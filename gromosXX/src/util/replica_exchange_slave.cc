/**
 * @file replica_exchange_slave.cc
 * replica exchange
 */

#include <stdheader.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <algorithm/algorithm/algorithm_sequence.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <algorithm/temperature/temperature_calculation.h>
#include <algorithm/temperature/berendsen_thermostat.h>

#include <io/argument.h>
#include <util/parse_verbosity.h>
#include <util/error.h>

#include <io/read_input.h>
#include <io/print_block.h>

#include <time.h>
#include <unistd.h>

#include <io/configuration/out_configuration.h>

#include "replica_exchange.h"

#undef MODULE
#undef SUBMODULE
#define MODULE util
#define SUBMODULE replica

#ifdef XXMPI


////////////////////////////////////////////////////////////////////////////////
// replica slave    ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

util::Replica_Exchange_Slave::Replica_Exchange_Slave()
{
}

int util::Replica_Exchange_Slave::run
(
 io::Argument & args)
{
  // create the simulation classes
  topology::Topology topo;
  configuration::Configuration conf;
  algorithm::Algorithm_Sequence md;
  simulation::Simulation sim;

  // read the files (could also be copied from the master)
  io::read_input(args, topo, conf, sim,  md, std::cout);

  // initialises everything
  md.init(topo, conf, sim, std::cout);

  // aliases
  std::vector<double> const & T = sim.param().replica.temperature;
  std::vector<double> const & l = sim.param().replica.lambda;

  io::Out_Configuration traj("GromosXX\n", std::cout);
  traj.title("GromosXX\n" + sim.param().title);
  traj.init(args["fin"], args["trj"], args["trv"], args["trf"], 
	    args["tre"], args["trg"], args["bae"], args["bag"],
	    sim.param());

  // should be the same as from master...
  std::cout << "\nMESSAGES FROM (SLAVE) INITIALIZATION\n";
  if (io::messages.display(std::cout) >= io::message::error){
    std::cout << "\nErrors during initialization!\n" << std::endl;
    MPI_Finalize();
    return 1;
  }
  io::messages.clear();
  
  std::cout.precision(5);
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);

  std::cout << "slave initialised" << std::endl;

  char port_name[MPI::MAX_PORT_NAME];

  if (args.count("slave") != 1){
    io::messages.add("slave: connection name required",
		     "replica exchange",
		     io::message::error);
    MPI_Finalize();
    return 1;
  }
  
  MPI::Lookup_name(args["slave"].c_str(), MPI_INFO_NULL, port_name);

  for(int run=0; run < sim.param().replica.slave_runs; ++run){
    
    DEBUG(8, "slave: connecting..");
    MPI_Comm_connect(port_name, MPI::INFO_NULL, 0, MPI::COMM_WORLD, &master);
    DEBUG(9, "slave: connected");

    // request a job
    int server_response = get_replica_data();

    if (server_response == 0){
      // accept job
      std::cout << "slave: got a job! (replica "
		<< replica_data.ID << ")" << std::endl;
	
      std::cout << "slave:  running replica " << replica_data.ID
		<< " at T=" << T[replica_data.Ti]
		<< " and l=" << l[replica_data.li]
		<< " (run = " << replica_data.run << ")"
		<< std::endl;

      // init replica parameters (t, conf, T, lambda)
      if (init_replica(topo, conf, sim)){
	MPI_Comm_disconnect(&master);
	MPI_Finalize();
	return 1;
      }
      

      // close connection
      MPI_Comm_disconnect(&master);

      // run it
      int error = run_md(topo, conf, sim, md, traj);

      if (!error){
	// do we need to reevaluate the potential energy ?
	// yes! 'cause otherwise it's just the energy for
	// the previous configuration...
	algorithm::Algorithm * ff = md.algorithm("Forcefield");
	
	if (ff == NULL){
	  std::cerr << "forcefield not found in MD algorithm sequence"
		    << std::endl;
	  MPI_Finalize();
	  MPI_Comm_free(&master);
	  return 1;
	}
	
	ff->apply(topo, conf, sim);
	conf.current().energies.calculate_totals();
	replica_data.epot_i = conf.current().energies.potential_total;
	
	std::cout << "replica_energy " << replica_data.epot_i 
		  << " @ " << T[replica_data.Ti] << "K" << std::endl;
	
	if (replica_data.li != replica_data.lj){
	  
	  // change the lambda value
	  sim.param().perturbation.lambda = replica_data.lj;
	  topo.lambda(replica_data.lj);
	  topo.update_for_lambda();
	  
	  // recalc energy
	  ff->apply(topo, conf, sim);
	  conf.current().energies.calculate_totals();
	  replica_data.epot_j = conf.current().energies.potential_total;
	} 
	else{
	  replica_data.epot_j = replica_data.epot_i;
	}
	
	++replica_data.run;
	replica_data.state = waiting;
      }
      else{
	replica_data.state = st_error;
      }
	
      DEBUG(8, "slave: connecting (after run)...");
      MPI_Comm_connect(port_name, MPI::INFO_NULL, 0, MPI::COMM_WORLD, &master);
      DEBUG(9, "slave: connected");

      DEBUG(8, "slave: finished job " << replica_data.ID);
      MPI_Send(&replica_data.ID, 1, MPI_INT, 0, 2, master);

      if (!error){
	// store configuration on master
	update_replica_data();
	update_configuration(topo, conf);
      }
      
      // and disconnect
      DEBUG(9, "disconnecting...");
      MPI_Comm_disconnect(&master);

    }
    else{
      MPI_Comm_disconnect(&master);
      std::cout << "slave waiting..." << std::endl;
      // wait apropriate time for master to prepare
      sleep(60);
    }
    
  } // for slave_runs
    
  // MPI_Comm_free(&master);
  
  return 0;
}

int util::Replica_Exchange_Slave::run_md
(
 topology::Topology & topo,
 configuration::Configuration & conf,
 simulation::Simulation & sim,
 algorithm::Algorithm_Sequence & md,
 io::Out_Configuration & traj
 )
{
  // do we scale the initial temperatures?
  if (sim.param().replica.scale){
    algorithm::Temperature_Calculation tcalc;
    tcalc.apply(topo, conf, sim);
    
    algorithm::Berendsen_Thermostat tcoup;
    tcoup.calc_scaling(topo, conf, sim, true);
    tcoup.scale(topo, conf, sim);
  }

  double end_time = sim.time() + 
    sim.time_step_size() * (sim.param().step.number_of_steps - 1);
    
  int error;

  while(sim.time() < end_time + math::epsilon){
      
    traj.write_replica_step(sim, replica_data);
    traj.write(conf, topo, sim, io::reduced);

    // run a step
    if ((error = md.run(topo, conf, sim))){
      
      std::cout << "\nError during MD run!\n" << std::endl;
      break;
    }

    traj.print(topo, conf, sim);

    sim.time() += sim.time_step_size();
    ++sim.steps();

  }
    
  return error;
}

int util::Replica_Exchange_Slave::get_replica_data()
{
  int i = 1;
  MPI_Status status;
  
  DEBUG(8, "slave: requesting job");
  MPI_Send(&i, 1, MPI_INT, 0, 1, master);
  
  DEBUG(8, "slave: waiting for replica data");
  MPI_Recv(&replica_data, sizeof(Replica_Data), MPI_CHAR,
	   MPI_ANY_SOURCE, MPI_ANY_TAG, master, &status);
  
  /*
  if (status.MPI_ERROR != MPI_SUCCESS){
    std::cout << "MPI ERROR! getting replica data: " << status.MPI_ERROR << std::endl;
    return 1;
  }
  */

  DEBUG(9, "slave: got replica " << replica_data.ID
	<< " temperature=" << replica_data.Ti
	<< " lambda=" << replica_data.li);
  
  return status.MPI_TAG;
}

int util::Replica_Exchange_Slave::update_replica_data()
{
  DEBUG(8, "slave: updating replica data");
  MPI_Send(&replica_data, sizeof(Replica_Data), MPI_CHAR,
	   0, 0, master);

  return 0;
}

int util::Replica_Exchange_Slave::get_configuration
(
 topology::Topology const & topo,
 configuration::Configuration & conf
 )
{
  int error = 0;
  
  MPI_Status status;

  DEBUG(10, "receiving " << 3 * conf.current().pos.size() << " coords");
  MPI_Recv(&conf.current().pos(0)(0), conf.current().pos.size() * 3,
	   MPI::DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, master, &status);

  /*
  if (status.MPI_ERROR != MPI_SUCCESS){
    std::cout << "MPI ERROR! (pos)" << status.MPI_ERROR << std::endl;
    error += 1;
  }
  */
  
  MPI_Recv(&conf.current().vel(0)(0), conf.current().vel.size()*3,
	   MPI::DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, master, &status);
  /*
  if (status.MPI_ERROR != MPI_SUCCESS){
    std::cout << "MPI ERROR! (vel)" << status.MPI_ERROR << std::endl;
    error += 2;
  }
  */
  
  MPI_Recv(&conf.current().box(0)(0), 9,
	   MPI::DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, master, &status);
  /*
  if (status.MPI_ERROR != MPI_SUCCESS){
    std::cout << "MPI ERROR! (box)" << status.MPI_ERROR << std::endl;
    error += 4;
  }
  */
  
  // return status.MPI_TAG;
  return error;
}

int util::Replica_Exchange_Slave::update_configuration
(
 topology::Topology const & topo,
 configuration::Configuration & conf
 )
{
  // positions
  DEBUG(9, "sending " << 3 * conf.current().pos.size() << " coords");
  
  MPI_Send(&conf.current().pos(0)(0), conf.current().pos.size()*3,
	   MPI::DOUBLE, 0, 0, master);
  
  // velocities
  MPI_Send(&conf.current().vel(0)(0), conf.current().vel.size()*3,
	   MPI::DOUBLE, 0, 0, master);

  // and box
  MPI_Send(&conf.current().box(0)(0), 9,
	   MPI::DOUBLE, 0, 0, master);
  
  return 0;
}

int util::Replica_Exchange_Slave::init_replica
(
 topology::Topology & topo,
 configuration::Configuration & conf,
 simulation::Simulation & sim
 )
{
  // get configuration from master
  if (get_configuration(topo, conf))
    return 1;
  
  // change all the temperature coupling temperatures
  for(unsigned int i=0; i<sim.multibath().size(); ++i){
    sim.multibath()[i].temperature = replica_data.Ti;
  }
  
  // change the lambda value
  sim.param().perturbation.lambda = replica_data.li;
  topo.lambda(replica_data.li);
  topo.update_for_lambda();
  
  // change simulation time
  sim.time() = replica_data.run * 
    sim.param().step.number_of_steps * sim.param().step.dt +
    sim.param().step.t0;
  
  sim.steps() = replica_data.run * sim.param().step.number_of_steps;

  return 0;
}

#endif

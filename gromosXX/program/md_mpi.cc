/**
 * @file md_mpi.cc
 * the main md program (MPI version)
 */

#ifdef XXMPI
#include <mpi.h>
#endif

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <algorithm/algorithm/algorithm_sequence.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <io/argument.h>
#include <util/parse_verbosity.h>
#include <util/usage.h>
#include <util/error.h>

#include <io/read_input.h>
#include <io/print_block.h>

#include <time.h>

#include <io/configuration/out_configuration.h>

#ifdef OMP
#include <omp.h>
#endif


#pragma hdrstop

#include "BUILD_NUMBER"

void print_title(bool color = false, int size = 1, std::ostream & os = std::cout);

int main(int argc, char *argv[]){

#ifdef XXMPI

  const double start = util::now();

  util::Known knowns;
  knowns << "topo" << "conf" << "input" << "verb" << "pttopo"
	 << "trj" << "fin" << "trv" << "trf" << "tre" << "trg"
	 << "bae" << "bag" << "posres" <<"distrest" << "jval"
	 << "print" << "version";
  
  
  std::string usage;
  util::get_usage(knowns, usage, argv[0]);
  usage += "#\n\n";

  // master or slave : that's the question
  MPI::Init(argc, argv);
  
  int rank, size;
  rank = MPI::COMM_WORLD.Get_rank();
  size = MPI::COMM_WORLD.Get_size();

  // create an output file (for the slaves)
  std::ostringstream oss;
  oss << "slave_" << rank << ".out";
  std::ofstream ofs(oss.str().c_str());
  
  bool quiet = false;
  std::ostream * os;
  if (rank == 0){
    os = &std::cout;
  }
  else{
    os = &ofs;
    quiet = true;
  }
  

  io::Argument args;

  if (args.parse(argc, argv, knowns)){
    if (rank == 0)
      std::cerr << usage << std::endl;
    MPI::Finalize();
    return 1;
  }
    
  if (args.count("version") >= 0){
    if (rank == 0)
      print_title(true, size);
    MPI::Finalize();
    return 0;
  }
  else print_title(false, size, *os);
    
  // parse the verbosity flag and set debug levels
  if (util::parse_verbosity(args)){
    if (rank == 0) std::cerr << "could not parse verbosity argument" << std::endl;
    MPI::Finalize();
    return 1;
  }

  // create the simulation classes
  topology::Topology topo;
  configuration::Configuration conf;
  algorithm::Algorithm_Sequence md;
  simulation::Simulation sim;

  // enable mpi for the nonbonded terms
  sim.mpi = true;
  
  if (io::read_input(args, topo, conf, sim,  md, *os, quiet)){
    io::messages.display(std::cout);
    std::cout << "\nErrors during initialization!\n" << std::endl;
    MPI::Finalize();
    return 1;    
  }
  
  // initialises all algorithms (and therefore also the forcefield)
  md.init(topo, conf, sim, *os, quiet);

  double end_time = sim.time() + 
    sim.time_step_size() * (sim.param().step.number_of_steps - 1);

  if(rank == 0){
    
    std::cout << "MPI master node (of " << size << " nodes)" << std::endl;
    
    io::Out_Configuration traj("GromosXX\n");
    traj.title("GromosXX\n" + sim.param().title);
    
    // create output files...
    traj.init(args, sim.param());
    
    std::cout << "\nMESSAGES FROM INITIALIZATION\n";
    if (io::messages.display(std::cout) >= io::message::error){
      // exit
      std::cout << "\nErrors during initialization!\n" << std::endl;
      return 1;
    }
    
    io::messages.clear();

    std::cout.precision(5);
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    
    std::cout << "\nenter the next level of molecular "
	      << "dynamics simulations\n" << std::endl;
    
    
    std::cout << "==================================================\n"
	      << " MAIN MD LOOP\n"
	      << "==================================================\n"
	      << std::endl;
    
    int error;
    
    const double init_time = util::now() - start;

    while(sim.time() < end_time + math::epsilon){
      
      traj.write(conf, topo, sim, io::reduced);
      
      // run a step
      if ((error = md.run(topo, conf, sim))){
	
	if (error == E_MINIMUM_REACHED){
	  conf.old().energies.calculate_totals();
	  traj.print_timestep(sim, traj.output());
	  io::print_ENERGY(traj.output(), conf.old().energies, 
			   topo.energy_groups(),
			   "MINIMUM ENERGY", "EMIN_");
	  
	  error = 0; // clear error condition
	  break;
	}
	
	std::cout << "\nError during MD run!\n" << std::endl;
	// try to save the final structures...
	break;
      }
      
      traj.print(topo, conf, sim);
      
      sim.time() += sim.time_step_size();
      ++sim.steps();
      
    }
    
    std::cout << "writing final configuration" << std::endl;
    
    traj.write(conf, topo, sim, io::final);
    traj.print_final(topo, conf, sim);

    std::cout << "\nMESSAGES FROM SIMULATION\n";
    io::message::severity_enum err_msg = io::messages.display(std::cout);
    
    std::cout << "\n\n";
    
    md.print_timing(std::cout);
    
    std::cout << "Overall time used:\t" << util::now() - start << "\n"
	      << "(initialization took " << init_time << ")\n\n";
    
    const time_t time_now = time_t(util::now());
    std::cout << ctime(&time_now) << "\n\n";
    
    if (error)
      std::cout << "\nErrors encountered during run - check above!\n" << std::endl;
    else if(err_msg > io::message::notice){
      std::cout << "\nGromosXX finished. "
		<< "Check the messages for possible problems during the run."
		<< std::endl;
    }
    else{
      std::cout << "\nGromosXX finished successfully\n" << std::endl;
    }
    
  }

  ////////////////////////////////////////////////////////////////////////////////
  // MPI Slave
  ////////////////////////////////////////////////////////////////////////////////

  else{
    (*os) << "MPI slave " << rank << " of " << size << std::endl;
    
    // let's get the forcefield
    interaction::Forcefield * ff = 
      dynamic_cast<interaction::Forcefield *>(md.algorithm("Forcefield"));
    
    if (ff == NULL){
      std::cerr << "MPI slave: could not access forcefield\n\t(internal error)" << std::endl;
      MPI::Finalize();
      return 1;
    }
    
    interaction::Interaction * nb = ff->interaction("NonBonded");
    if (nb == NULL){
      std::cerr << "MPI slave: could not get NonBonded interactions from forcefield"
		<< "\n\t(internal error)"
		<< std::endl;
      MPI::Finalize();
      return 1;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // run the simulation
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    const double init_time = util::now() - start;
    int error;

    while(sim.time() < end_time + math::epsilon){
      // run a step
      // (*os) << "waiting for master (nonbonded interaction)" << std::endl;
      // DEBUG(10, "slave " << rank << " waiting for master");
      if ((error = nb->calculate_interactions(topo, conf, sim)) != 0){
	std::cout << "MPI slave " << rank << ": error in nonbonded calculation!\n" << std::endl;
	break;
      }
      
      // DEBUG(10, "slave " << rank << " step done");
      // (*os) << "step done (it really worked?)" << std::endl;

      sim.time() += sim.time_step_size();
      ++sim.steps();
    }

    std::cout << "\nMESSAGES FROM SIMULATION\n";
    io::message::severity_enum err_msg = io::messages.display(std::cout);
    
    std::cout << "\n\n";
    
    md.print_timing(std::cout);
    
    std::cout << "Overall time used:\t" << util::now() - start << "\n"
	      << "(initialization took " << init_time << ")\n\n";
    
    const time_t time_now = time_t(util::now());
    std::cout << ctime(&time_now) << "\n\n";
    
    if (error)
      std::cout << "\nErrors encountered during run - check above!\n" << std::endl;
    else if(err_msg > io::message::notice){
      std::cout << "\nGromosXX MPI slave " << rank << " finished. "
		<< "Check the messages for possible problems during the run."
		<< std::endl;
    }
    else{
      std::cout << "\nGromosXX MPI slave " << rank << " finished successfully\n" << std::endl;
    }
    
  } // end of slave

  os = NULL;
  
  ofs.flush();
  ofs.close();

  // and exit...
  MPI::Finalize();
  return 0;

#else
  std::cout << argv[0] << " needs MPI to run\n\tuse --enable-mpi at configure\n" << std::endl;
  return 1;
#endif

}

////////////////////////////////////////////////////////////////////////////////
// helper functions
////////////////////////////////////////////////////////////////////////////////

void print_title(bool color, int size, std::ostream & os)
{
  if (color){

#ifdef NDEBUG
#ifndef BZDEBUG
    os << "\033[1;32m";
#else
    os << "\033[1;31m";
#endif
#else
    os << "\033[1;31m";
#endif
    os << "\n\nGromosXX 0.2.1 development\033[22;0m\n\n"
       << "26th October 2004\n";
  }
  else
    os << "\n\nGromosXX 0.2.1 development\n\n"
       << "26th October 2004\n";
  
  os << "build date    " << BUILD_DATE << "\n"
     << "build number  " << BUILD_NUMBER << "\n\n";
  
#ifdef NDEBUG
  os << "standard library debugging disabled.\n";
#else
  os << "standard library debugging enabled.\n";
#endif

  // some omp stuff
#ifdef OMP
  int nthreads, tid;
#pragma omp parallel private(nthreads, tid)
  {
    tid = omp_get_thread_num();
    if (tid == 0){
      nthreads = omp_get_num_threads();
      os << "OpenMP code enabled\n"
	 << "\tshared memory parallelization\n"
	 << "\twww.openmp.org\n\n"
	 << "\tusing "
	 << omp_get_num_threads() << " threads\n"
	 << "\tthis can be adjusted by setting the\n"
	 << "\tOMP_NUM_THREADS environment variable\n"
	 << std::endl;
    }
    
  }
#endif
  
  os << "MPI code enabled\n"
     << "\tdistributed memory parallelization\n"
     << "\twww.mpi-forum.org\n\n";
  if (size > 1)
    os << "\trunning on " << size << " nodes\n";
  else
    os << "\trunning on " << size << " node\n";
  
  os << "\nGruppe fuer Informatikgestuetzte Chemie\n"
     << "Professor W. F. van Gunsteren\n"
     << "Swiss Federal Institute of Technology\n"
     << "Zuerich\n\n"
     << "Bugreports to http://www.igc.ethz.ch:5555\n"
     << std::endl;
  
}

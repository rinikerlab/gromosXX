/**
 * @file md_mpi.cc
 * the main md program (MPI version)
 */
/**
 * @page programs Program Documentation
 *
 * @anchor md_mpi
 * @section md_mpi parallel molecular dynamics with MPI
 * @date 28.10.2008
 *
 * Program md_mpi is used to run parallel molecular dynamics simulations using 
 * MPI. In order to use this program one has to compile GROMOS with MPI 
 * support:
 * @verbatim
$ ../configure --enable-mpi CC=mpicc CXX=mpiCC
@endverbatim
 *
 * See @ref md for the documentation of the command line arguments
 */

#ifdef XXMPI
#include <mpi.h>
#endif
#include <math/fft.h>

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <algorithm/algorithm/algorithm_sequence.h>
#include <math/periodicity.h>
#include <algorithm/constraints/shake.h>
#include <algorithm/constraints/m_shake.h>
#include <algorithm/integration/monte_carlo.h>
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

#include <sys/utsname.h>

#pragma hdrstop

#include "BUILD_NUMBER"

void print_title(bool color = false, int size = 1, std::ostream & os = std::cout);

int main(int argc, char *argv[]){

#ifdef XXMPI

  const double start = util::now();

  util::Known knowns;
  knowns << "topo" << "conf" << "input" << "verb" << "pttopo"
	 << "trj" << "fin" << "trv" << "trf" << "trs" << "tramd" << "tre" << "trg"
	 << "bae" << "bag" << "posresspec" << "refpos" << "distrest" 
         << "dihrest" << "jval" << "xray" << "lud" << "led" << "anatrj"
         << "print" << "friction" << "version";
  
  
  std::string usage;
  util::get_usage(knowns, usage, argv[0]);
  usage += "#\n\n";

  // master or slave : that's the question
  MPI::Init(argc, argv);
  FFTW3(mpi_init());
  
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
    FFTW3(mpi_cleanup());
    MPI::Finalize();
    return 1;
  }
    
  if (args.count("version") >= 0){
    if (rank == 0)
      print_title(true, size);
    MPI::Finalize();
    FFTW3(mpi_cleanup());
    return 0;
  }
  else print_title(false, size, *os);
    
  // parse the verbosity flag and set debug levels
  if (util::parse_verbosity(args)){
    if (rank == 0) std::cerr << "could not parse verbosity argument" << std::endl;
    FFTW3(mpi_cleanup());
    MPI::Finalize();
    return 1;
  }

  // create the simulation classes
  topology::Topology topo;
  configuration::Configuration conf;
  simulation::Simulation sim;
  algorithm::Algorithm_Sequence md;

  // enable mpi for the nonbonded terms
  sim.mpi = true;
  
  if (io::read_input(args, topo, conf, sim,  md, *os, quiet)){
    io::messages.display(std::cout);
    std::cout << "\nErrors during initialization!\n" << std::endl;
    FFTW3(mpi_cleanup());
    MPI::Finalize();
    return 1;    
  }
  
  // initialises all algorithms (and therefore also the forcefield)
  md.init(topo, conf, sim, *os, quiet);

  
  int error;

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
      MPI::Finalize();
      return 1;
    }
    
    io::messages.clear();

    std::cout.precision(5);
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    
    std::cout << "\nenter the next level of molecular "
	      << "dynamics simulations\n" << std::endl;
    
    
    int percent = 0;
    
    std::cout << "==================================================\n"
	      << " MAIN MD LOOP\n"
	      << "==================================================\n"
	      << std::endl;
    
    const double init_time = util::now() - start;

    int next_step = 1;

    while(int(sim.steps()) < sim.param().step.number_of_steps){
      
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
        // send error status to slaves
        next_step = 0;
        std::cout << "Telling slaves to quit." << std::endl;
        MPI::COMM_WORLD.Bcast(&next_step, 1, MPI::INT, 0);

	// try to save the final structures...
	break;
      }

      // tell the slaves to continue
      MPI::COMM_WORLD.Bcast(&next_step, 1, MPI::INT, 0);
      
      traj.print(topo, conf, sim);
      
      ++sim.steps();
      sim.time() = sim.param().step.t0 + sim.steps()*sim.time_step_size();
      
      
      if ((sim.param().step.number_of_steps / 10 > 0) &&
	  (sim.steps() % (sim.param().step.number_of_steps / 10) == 0)){
        ++percent;
        const double spent = util::now() - start;
        const int hh = int(spent / 3600);
        const int mm = int((spent - hh * 3600) / 60);
        const int ss = int(spent - hh * 3600 - mm * 60);

        std::cerr << "MD:       " << std::setw(4) << percent * 10 << "% done..." << std::endl;
        std::cout << "MD:       " << std::setw(4) << percent * 10 << "% done..." << std::endl;
        std::cerr << "MD: spent " << hh << ":" << mm << ":" << ss << std::endl;
        std::cout << "MD: spent " << hh << ":" << mm << ":" << ss << std::endl;

        const double eta_spent = spent / sim.steps() * sim.param().step.number_of_steps - spent;
        const int eta_hh = int(eta_spent / 3600);
        const int eta_mm = int((eta_spent - eta_hh * 3600) / 60);
        const int eta_ss = int(eta_spent - eta_hh * 3600 - eta_mm * 60);
      
        std::cerr << "MD: ETA   " << eta_hh << ":" << eta_mm << ":" << eta_ss << std::endl;
        std::cout << "MD: ETA   " << eta_hh << ":" << eta_mm << ":" << eta_ss << std::endl;
      }
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
    
    // check whether we have nonbonded interactions
    bool do_nonbonded = (sim.param().force.nonbonded_crf || 
                         sim.param().force.nonbonded_vdw);
    
    // let's get the forcefield
    interaction::Forcefield * ff = 
      dynamic_cast<interaction::Forcefield *>(md.algorithm("Forcefield"));
    
    if (do_nonbonded && ff == NULL){
      std::cerr << "MPI slave: could not access forcefield\n\t(internal error)" << std::endl;
      MPI::Finalize();
      return 1;
    }
    
    interaction::Interaction * nb = ff->interaction("NonBonded");
    if (do_nonbonded && nb == NULL){
      std::cerr << "MPI slave: could not get NonBonded interactions from forcefield"
		<< "\n\t(internal error)"
		<< std::endl;
      MPI::Finalize();
      return 1;
    }

    // get shake and check whether we do it for solvent
    bool do_shake = sim.param().system.nsm &&
      sim.param().constraint.solvent.algorithm == simulation::constr_shake;

    algorithm::Shake * shake =
      dynamic_cast<algorithm::Shake *>(md.algorithm("Shake"));
    if (do_shake && shake == NULL) {
        std::cerr << "MPI slave: could not get Shake algorithm from MD sequence."
                << "\n\t(internal error)"
                << std::endl;
      MPI::Finalize();
      return 1;
    }

    // get m_shake and check whether we do it for solvent
    bool do_m_shake = sim.param().system.nsm &&
      sim.param().constraint.solvent.algorithm == simulation::constr_m_shake;

    algorithm::M_Shake * m_shake =
      dynamic_cast<algorithm::M_Shake *>(md.algorithm("M_Shake"));
    if (do_m_shake && m_shake == NULL) {
        std::cerr << "MPI slave: could not get M_Shake algorithm from MD sequence."
                << "\n\t(internal error)"
                << std::endl;
      MPI::Finalize();
      return 1;
    }
    
    // get chemical monte carlo
    bool do_cmc = sim.param().montecarlo.mc;
    
    algorithm::Monte_Carlo * monte_carlo = 
            dynamic_cast<algorithm::Monte_Carlo *>(md.algorithm("MonteCarlo"));
    if(do_cmc && monte_carlo == NULL){
      std::cerr << "MPI slave: could not get Monte Carlo algorithm from MD sequence."
              << "\n\t(internal error)"
              << std:: endl;
      MPI::Finalize();
      return 1;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // run the simulation
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    const double init_time = util::now() - start;
    int next_step = 0 ;

    while(int(sim.steps()) < sim.param().step.number_of_steps){
      // run a step
      // (*os) << "waiting for master (nonbonded interaction)" << std::endl;
      // DEBUG(10, "slave " << rank << " waiting for master");
      if (do_nonbonded && (error = nb->calculate_interactions(topo, conf, sim)) != 0){
	std::cout << "MPI slave " << rank << ": error in nonbonded calculation!\n" << std::endl;
      }
      
      // DEBUG(10, "slave " << rank << " step done");
      // (*os) << "step done (it really worked?)" << std::endl;
      
      if (do_cmc && (error = monte_carlo->apply(topo, conf, sim)) != 0){
        std::cout << "MPI slave " << rank << ": error in Monte Carlo algorithm!\n" 
                << std::endl;
      }

      if (do_shake && (error = shake->apply(topo, conf, sim)) != 0) {
        std::cout << "MPI slave " << rank << ": error in Shake algorithm!\n" << std::endl;
      }

      if (do_m_shake && (error = m_shake->apply(topo, conf, sim)) != 0) {
        std::cout << "MPI slave " << rank << ": error in M-Shake algorithm!\n" << std::endl;
      }
 

      MPI::COMM_WORLD.Bcast(&next_step, 1, MPI::INT, 0);

      if (!next_step) {
        (*os) << "There was an error in the master. Check output file for details." << std::endl
              << "Exiting from MD main loop." << std::endl;
        error = 1;
        break;
      }

      ++sim.steps();
      sim.time() = sim.param().step.t0 + sim.steps()*sim.time_step_size();
    }

    (*os) << "\nMESSAGES FROM SIMULATION\n";
    io::message::severity_enum err_msg = io::messages.display(*os);
    
    (*os) << "\n\n";
    
    md.print_timing(*os);
    
    (*os) << "Overall time used:\t" << util::now() - start << "\n"
	  << "(initialization took " << init_time << ")\n\n";
    
    const time_t time_now = time_t(util::now());
    (*os) << ctime(&time_now) << "\n\n";
    
    if (error)
      (*os) << "\nErrors encountered during run - check above!\n" << std::endl;
    else if(err_msg > io::message::notice){
      (*os) << "\nGromosXX MPI slave " << rank << " finished. "
	    << "Check the messages for possible problems during the run."
	    << std::endl;
    }
    else{
      (*os) << "\nGromosXX MPI slave " << rank << " finished successfully\n" << std::endl;
    }
    
  } // end of slave

  os = NULL;
  
  ofs.flush();
  ofs.close();

  // and exit...
  FFTW3(mpi_cleanup());
  MPI::Finalize();
  return error;

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
    os << "\n\nGromosXX 0.3.0 development\033[22;0m\n\n"
       << "12 February 2008\n";
  }
  else
    os << "\n\nGromosXX 0.3.0 development\n\n"
       << "12 February 2008\n";
  
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
     << "Bugreports to https://gromos/svn/trac/gromosXXc++\n"
     << std::endl;

  struct utsname sysinf;
  if (uname(&sysinf) != -1){
    std::cout << "running on"
	      << "\n\t" << sysinf.nodename
	      << "\n\t" << sysinf.sysname
	      << " " << sysinf.release
	      << " " << sysinf.version
	      << " " << sysinf.machine
	      << "\n\n";
  }
}

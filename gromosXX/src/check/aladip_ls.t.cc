/**
 * @file aladip.t.cc
 * tests using aladip
 */


#include "../stdheader.h"

#include "../algorithm/algorithm.h"
#include "../topology/topology.h"
#include "../simulation/simulation.h"
#include "../configuration/configuration.h"

#include "../algorithm/algorithm/algorithm_sequence.h"
#include "../interaction/interaction.h"
#include "../interaction/forcefield/forcefield.h"

#include "../io/argument.h"
#include "../util/parse_verbosity.h"
#include "../util/usage.h"
#include "../util/error.h"

#include "../interaction/interaction_types.h"
#include "../io/instream.h"
#include "../util/parse_tcouple.h"
#include "../io/blockinput.h"
#include "../io/topology/in_topology.h"

#include "../algorithm/integration/leap_frog.h"
#include "../algorithm/temperature/temperature_calculation.h"
#include "../algorithm/temperature/berendsen_thermostat.h"
#include "../algorithm/pressure/pressure_calculation.h"
#include "../algorithm/pressure/berendsen_barostat.h"

#include "../interaction/forcefield/create_forcefield.h"

#include "../util/create_simulation.h"
#include "../algorithm/create_md_sequence.h"

#include <time.h>

#include "check.h"

#include "check_forcefield.h"
#include "check_state.h"

void hard_coded_values(std::map<std::string, double> & m){
  m["NonBonded_-1"] = -168.832;
  m["NonBonded_0"]  = -129.245;
  m["NonBonded_1"]  = -125.946;
  m["NonBonded_2"]  = -129.641;
  m["NonBonded_3"]  = -123.305;
  m["NonBonded_4"]  = -124.849;
  m["NonBonded_5"]  = -130.201;
  m["NonBonded_6"]  = -125.314;
  m["NonBonded_7"]  = -130.103;
  m["NonBonded_8"]  = -137.76;
  m["NonBonded_9"]  = -132.338;
  m["NonBonded_10"] = -139.645;
}

#ifdef OMP
  #include <omp.h>
#endif

int main(int argc, char* argv[]) {

#ifdef OMP
  omp_set_num_threads(1);
#endif

  int total = 0;
  
  util::Known knowns;
  knowns << "topo" << "pttopo" << "conf" << "input" << "verb";

  std::string usage = argv[0];
  usage += "\n\t[@topo    <topology>]\n";
  usage += "\t[@pttopo <perturbation topology>]\n";
  usage += "\t[@conf    <starting configuration>]\n";
  usage += "\t[@input   <input>]\n";
  usage += "\t[@verb   <[module:][submodule:]level>]\n";

  io::Argument args;
  if (args.parse(argc, argv, knowns, true)){
    std::cerr << usage << std::endl;
    return 1;
  }

  // parse the verbosity flag and set debug levels
  util::parse_verbosity(args);
      
    std::string stopo, spttopo, sconf, sinput;
  bool quiet = true;

  if (args.count("verb") != -1) quiet = false;
      
  if(args.count("topo") == 1)
    stopo = args["topo"];
  else
    GETFILEPATH(stopo, "aladip.topo", "src/check/data/");

  if(args.count("pttopo") == 1)
    spttopo = args["pttopo"];
  else
    GETFILEPATH(spttopo, "aladip.pttopo", "src/check/data/");
      
  if(args.count("conf") == 1)
    sconf = args["conf"];
  else
    GETFILEPATH(sconf, "aladip.conf", "src/check/data/");

  if(args.count("input") == 1)
    sinput = args["input"];
  else
    GETFILEPATH(sinput, "aladip_ls.in", "src/check/data/");

  if (!quiet)
    std::cout << "\n\n"
	      << "topology :      " << stopo << "\n"
	      << "perturbation :  " << spttopo << "\n"
	      << "input :         " << sinput << "\n"
	      << "configuration : " << sconf << "\n"
	      << std::endl;

  // set hard coded values to compare to
  std::map<std::string, double> ref_values;
  hard_coded_values(ref_values);
  
  util::simulation_struct aladip_sim;
  io::In_Topology in_topo;

  in_topo.quiet = quiet;
      
  if (util::create_simulation(stopo,
			      spttopo,
			      sconf,
			      sinput,
			      aladip_sim,
			      in_topo,
			      "", "", "", "", "", 
			      quiet
			      )
      != 0){
    std::cerr << "creating simulation failed!" << std::endl;
    return 1;
  }
      
  if (true){
    // create a forcefield
    interaction::Forcefield *ff = new interaction::Forcefield;
	
    if (interaction::create_g96_forcefield(*ff, 
					   aladip_sim.topo,
					   aladip_sim.sim,
					   in_topo,
					   std::cout,
					   quiet)
	!= 0){
      std::cerr << "creating forcefield failed!" << std::endl;
      return 1;
    }

    std::cout << "Checking Ewald. This can take some time...\n";
    aladip_sim.sim.param().nonbonded.method = simulation::el_ewald;
    aladip_sim.sim.param().nonbonded.ls_charge_shape = -1;
    ff->init(aladip_sim.topo, aladip_sim.conf, aladip_sim.sim, std::cout,  quiet);
    // first check the forcefield
    total += check::check_forcefield(aladip_sim.topo, aladip_sim.conf,
				     aladip_sim.sim, *ff, ref_values);

    aladip_sim.sim.param().nonbonded.method = simulation::el_p3m;
    ff->init(aladip_sim.topo, aladip_sim.conf, aladip_sim.sim, std::cout,  quiet);

    std::cout << "Checking P3M. This can take a long time...\n";
    // first check the forcefield
    for (int shape = -1; shape < 11; ++shape) {
      std::cout << "Charge shape type: " << shape << "\n";
      aladip_sim.sim.param().nonbonded.ls_charge_shape = shape;
      total += check::check_forcefield(aladip_sim.topo, aladip_sim.conf,
              aladip_sim.sim, *ff, ref_values);

      // check virial, ...
      if (shape == -1 || shape == 3)
      total += check::check_state(aladip_sim.topo, aladip_sim.conf,
              aladip_sim.sim, *ff);
    }
  }
  // vtune: run the thing...
  else{
	
    algorithm::create_md_sequence(aladip_sim.md,
				  aladip_sim.topo,
				  aladip_sim.sim,
				  in_topo);
    
    double end_time = aladip_sim.sim.param().step.t0 + 
      aladip_sim.sim.time_step_size() * 
      aladip_sim.sim.param().step.number_of_steps;
    
    int error;

    while(aladip_sim.sim.time() < end_time){
      
      if ((error = aladip_sim.md.run(aladip_sim.topo, 
				     aladip_sim.conf,
				     aladip_sim.sim))){

	std::cout << "\nError during MD run!\n" << std::endl;
	// try to save the final structures...
	break;
      }

      // update the energies
      aladip_sim.conf.old().energies.calculate_totals();
      // perturbed energy derivatives
      if (aladip_sim.sim.param().perturbation.perturbation){
	aladip_sim.conf.old().perturbed_energy_derivatives.calculate_totals();
      }

      aladip_sim.conf.current().averages.
	apply(aladip_sim.topo,
	      aladip_sim.conf,
	      aladip_sim.sim);
	  
      aladip_sim.sim.time() +=  aladip_sim.sim.time_step_size();
      ++ aladip_sim.sim.steps();

    }
    aladip_sim.md.print_timing(std::cout);
  }

  return total;
}

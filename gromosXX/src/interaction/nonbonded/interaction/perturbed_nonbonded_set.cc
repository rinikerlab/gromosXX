/**
 * @file perturbed_nonbonded_set.cc
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <interaction/interaction_types.h>
#include <interaction/nonbonded/interaction/nonbonded_parameter.h>

#include <interaction/nonbonded/pairlist/pairlist.h>
#include <interaction/nonbonded/pairlist/pairlist_algorithm.h>

#include <math/periodicity.h>

#include <interaction/nonbonded/interaction/storage.h>
#include <interaction/nonbonded/interaction/nonbonded_outerloop.h>
#include <interaction/nonbonded/interaction/perturbed_nonbonded_outerloop.h>

#include <interaction/nonbonded/interaction/nonbonded_term.h>
#include <interaction/nonbonded/interaction/perturbed_nonbonded_term.h>

#include <interaction/nonbonded/interaction/perturbed_nonbonded_pair.h>

#include <math/volume.h>

#include <interaction/nonbonded/interaction/nonbonded_set.h>
#include <interaction/nonbonded/interaction/perturbed_nonbonded_set.h>

#include <util/debug.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE nonbonded

/**
 * Constructor.
 */
interaction::Perturbed_Nonbonded_Set
::Perturbed_Nonbonded_Set(Pairlist_Algorithm & pairlist_alg, Nonbonded_Parameter & param)
  : Nonbonded_Set(pairlist_alg, param),
    m_perturbed_outerloop(param),
    m_perturbed_pair(param)
{
}

/**
 * calculate nonbonded forces and energies.
 */
int interaction::Perturbed_Nonbonded_Set
::calculate_interactions(topology::Topology & topo,
			 configuration::Configuration & conf,
			 simulation::Simulation & sim,
			 int tid, int num_threads)
{
  DEBUG(4, "Nonbonded_Set::calculate_interactions");

  const double l = topo.lambda();
  
  if (sim.param().perturbation.scaling){
    // calculate lambda primes and d lambda prime / d lambda derivatives
    std::map<std::pair<int, int>, std::pair<int, double> >::const_iterator
      it = topo.energy_group_lambdadep().begin(),
      to = topo.energy_group_lambdadep().end();
    
    for(unsigned int i=0; it != to; ++i, ++it){
      
      const double alpha = it->second.second;
      double lp = alpha * l * l + (1-alpha) * l;
      double dlp = (2 * l - 1.0) * alpha + 1;
      
      // some additional flexibility
      if (lp > 1.0) {
	lp = 1.0;
	dlp = 0.0;
      }
      
      else if (lp < 0.0){
	lp = 0.0;
	dlp = 0.0;
      }
      
      // -1 or not???
      assert(int(topo.lambda_prime().size()) > it->second.first);
      assert(int(topo.lambda_prime_derivative().size()) > it->second.first);
      assert(it->second.first >= 0);
      
      topo.lambda_prime()[it->second.first] = lp;
      topo.lambda_prime_derivative()[it->second.first] = dlp;
      
    }
  }
    
  // zero forces, energies, virial...
  m_shortrange_storage.zero();

  // need to update pairlist?
  if(!(sim.steps() % sim.param().pairlist.skip_step)){
    DEBUG(7, "\tdoing longrange...");
    
    //====================
    // create a pairlist
    //====================
    
    // zero the longrange forces, energies, virial
    m_longrange_storage.zero();

    // parallelisation using STRIDE:
    // chargegroup based pairlist can only use this one!!!!
    // TODO:
    // move decision to pairlist!!!
    m_pairlist_alg.update_perturbed(topo, conf, sim, 
				    longrange_storage(),
				    pairlist(), perturbed_pairlist(),
				    tid, topo.num_atoms(), num_threads);

    /*
    sleep(2*tid);
    
    std::cout << "PRINTING OUT THE PAIRLIST\n\n";
    for(unsigned int i=0; i<100; ++i){
      if (i >= pairlist().size()) break;

      std::cout << "\n\n--------------------------------------------------";
      std::cout << "\n" << i;
      for(unsigned int j=0; j<pairlist()[i].size(); ++j){

	if (j % 10 == 0) std::cout << "\n\t";
	std::cout << std::setw(7) << pairlist()[i][j];
      }
    }
    */
  }

  // calculate forces / energies
  DEBUG(7, "\tshort range interactions");

  m_outerloop.lj_crf_outerloop(topo, conf, sim,
			       m_pairlist, m_shortrange_storage);

  DEBUG(7, "\tperturbed short range");
  m_perturbed_outerloop.perturbed_lj_crf_outerloop(topo, conf, sim, 
						   m_perturbed_pairlist,
						   m_shortrange_storage);
  // add 1,4 - interactions
  if (tid == 0){
    DEBUG(7, "\t1,4 - interactions");
    m_outerloop.one_four_outerloop(topo, conf, sim, m_shortrange_storage);

    DEBUG(7, "\tperturbed 1,4 - interactions");
    m_perturbed_outerloop.perturbed_one_four_outerloop(topo, conf, sim, 
						       m_shortrange_storage);
  
    // possibly do the RF contributions due to excluded atoms
    if(sim.param().longrange.rf_excluded){
      DEBUG(7, "\tRF excluded interactions and self term");
      m_outerloop.RF_excluded_outerloop(topo, conf, sim, m_shortrange_storage);

      DEBUG(7, "\tperturbed RF excluded interactions and self term");
      m_perturbed_outerloop.perturbed_RF_excluded_outerloop(topo, conf, sim,
							    m_shortrange_storage);

    }

    DEBUG(7, "\tperturbed pairs");
    m_perturbed_pair.perturbed_pair_outerloop(topo, conf, sim, m_shortrange_storage);

  }
  
  // add long-range force
  DEBUG(7, "\t(set) add long range forces");

  m_shortrange_storage.force += m_longrange_storage.force;
  
  // and long-range energies
  DEBUG(7, "\t(set) add long range energies");
  const unsigned int lj_e_size = unsigned(m_shortrange_storage.energies.lj_energy.size());
  
  for(unsigned int i = 0; i < lj_e_size; ++i){
    for(unsigned int j = 0; j < lj_e_size; ++j){
      m_shortrange_storage.energies.lj_energy[i][j] += 
	m_longrange_storage.energies.lj_energy[i][j];
      m_shortrange_storage.energies.crf_energy[i][j] += 
	m_longrange_storage.energies.crf_energy[i][j];
    }
  }

  // add longrange virial
  if (sim.param().pcouple.virial){
    DEBUG(7, "\t(set) add long range virial");
    for(unsigned int i=0; i<3; ++i){
      for(unsigned int j=0; j<3; ++j){

	DEBUG(8, "longrange virial = " << m_longrange_storage.virial_tensor(i,j)
	      << "\tshortrange virial = " << m_shortrange_storage.virial_tensor(i,j));

	m_shortrange_storage.virial_tensor(i,j) +=
	  m_longrange_storage.virial_tensor(i,j);
      }
    }
  }
  
  // and long-range energy lambda-derivatives
  DEBUG(7, "(set) add long-range lambda-derivatives");
  
  const unsigned int lj_size 
    = unsigned(m_shortrange_storage.perturbed_energy_derivatives.lj_energy.size());
  
  for(unsigned int i = 0; i < lj_size; ++i){
    for(unsigned int j = 0; j < lj_size; ++j){
      
      assert(m_shortrange_storage.perturbed_energy_derivatives.
	     lj_energy.size() > i);
      assert(m_shortrange_storage.perturbed_energy_derivatives.
	     lj_energy[i].size() > j);
      assert(m_shortrange_storage.perturbed_energy_derivatives.
	     lj_energy.size() > j);
      assert(m_shortrange_storage.perturbed_energy_derivatives.
	     lj_energy[j].size() > i);
      
      m_shortrange_storage.perturbed_energy_derivatives.lj_energy[i][j] += 
	m_longrange_storage.perturbed_energy_derivatives.lj_energy[i][j];
      
      m_shortrange_storage.perturbed_energy_derivatives.crf_energy[i][j] += 
	m_longrange_storage.perturbed_energy_derivatives.crf_energy[i][j];
      
    }
  }

  return 0;
}

/**
 * calculate the hessian for a given atom.
 * this will be VERY SLOW !
 */
int interaction::Perturbed_Nonbonded_Set
::calculate_hessian(topology::Topology & topo,
		    configuration::Configuration & conf,
		    simulation::Simulation & sim,
		    unsigned int atom_i, unsigned int atom_j,
		    math::Matrix & hessian){
  
  if (topo.is_perturbed(atom_i) ||
      topo.is_perturbed(atom_j)){
    assert(false);
    return -1;
  }

  return Nonbonded_Set::calculate_hessian(topo, conf, sim,
					  atom_i, atom_j, hessian);
}

int interaction::Perturbed_Nonbonded_Set
::init(topology::Topology const & topo,
       configuration::Configuration const & conf,
       simulation::Simulation const & sim,
       bool quiet)
{
  Nonbonded_Set::init(topo, conf, sim, quiet);
  
  m_shortrange_storage.perturbed_energy_derivatives.resize
    (unsigned(conf.current().perturbed_energy_derivatives.bond_energy.size()),
     unsigned(conf.current().perturbed_energy_derivatives.kinetic_energy.size()));

  m_longrange_storage.perturbed_energy_derivatives.resize
    (unsigned(conf.current().perturbed_energy_derivatives.bond_energy.size()),
     unsigned(conf.current().perturbed_energy_derivatives.kinetic_energy.size()));

  perturbed_pairlist().resize(topo.num_atoms());

  return 0;
}


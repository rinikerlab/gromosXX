/**
 * @file standard_pairlist_algorithm.cc
 * standard pairlist algorithm
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <interaction/interaction_types.h>
#include <math/periodicity.h>

#include <interaction/nonbonded/pairlist/pairlist.h>
#include <interaction/nonbonded/interaction/storage.h>
#include <interaction/nonbonded/interaction/nonbonded_parameter.h>

#include <interaction/nonbonded/interaction/nonbonded_term.h>
#include <interaction/nonbonded/interaction/nonbonded_innerloop.h>

#include <interaction/nonbonded/interaction/perturbed_nonbonded_term.h>
#include <interaction/nonbonded/interaction/perturbed_nonbonded_innerloop.h>

#include <interaction/nonbonded/pairlist/pairlist_algorithm.h>
#include <interaction/nonbonded/pairlist/standard_pairlist_algorithm.h>

#include <interaction/nonbonded/interaction_spec.h>

#include <util/debug.h>
#include <util/template_split.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE pairlist

void interaction::Standard_Pairlist_Algorithm::
update_perturbed(topology::Topology & topo,
		 configuration::Configuration & conf,
		 simulation::Simulation & sim,
		 interaction::Storage & storage,
		 interaction::Pairlist & pairlist,
		 interaction::Pairlist & perturbed_pairlist,
		 unsigned int begin, unsigned int end,
		 unsigned int stride)
{
  if (sim.param().pairlist.atomic_cutoff){
    SPLIT_PERTURBATION(update_pert_atomic,
		       topo, conf, sim, storage,
		       pairlist, perturbed_pairlist,
		       begin, end, stride);
  }
  else{
    SPLIT_PERTURBATION(update_pert_cg,
		       topo, conf, sim, storage,
		       pairlist, perturbed_pairlist, 
		       begin, end, stride);
  }
}

template<typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm::
update_pert_cg(topology::Topology & topo,
	       configuration::Configuration & conf,
	       simulation::Simulation & sim, 
	       interaction::Storage & storage,
	       interaction::Pairlist & pairlist,
	       interaction::Pairlist & perturbed_pairlist,
	       unsigned int begin, unsigned int end,
	       unsigned int stride)
{
  SPLIT_PERT_INNERLOOP(_update_pert_cg,
		       topo, conf, sim, storage,
		       pairlist, perturbed_pairlist, 
		       begin, end, stride);
}

template<typename t_interaction_spec, typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm::
_update_pert_cg(topology::Topology & topo,
		configuration::Configuration & conf,
		simulation::Simulation & sim,
		interaction::Storage & storage,
		interaction::Pairlist & pairlist,
		interaction::Pairlist & perturbed_pairlist,
		unsigned int begin, unsigned int end,
		unsigned int stride)
{
  DEBUG(7, "standard pairlist update");
  const double update_start = util::now();
  
  // create the innerloops
  math::Periodicity<t_interaction_spec::boundary_type> periodicity(conf.current().box);
  
  Nonbonded_Innerloop<t_interaction_spec> innerloop(*m_param);
  innerloop.init(sim);
  
  Perturbed_Nonbonded_Innerloop<t_interaction_spec, t_perturbation_details>
    perturbed_innerloop(*m_param);
  perturbed_innerloop.init(sim);
  perturbed_innerloop.set_lambda(topo.lambda(), topo.lambda_exp());

  // empty the pairlist
  assert(pairlist.size() == topo.num_atoms());
  assert(perturbed_pairlist.size() == topo.num_atoms());
  
  for(unsigned int i=0; i<topo.num_atoms(); ++i){
    pairlist[i].clear();
    perturbed_pairlist[i].clear();
  }

  DEBUG(7, "pairlist cleared");
  
  // loop over the chargegroups
  const int num_cg = topo.num_chargegroups();
  const int num_solute_cg = topo.num_solute_chargegroups();
  int cg1_index, cg1_to;

  DEBUG(8, "num cg        = " << num_cg);
  DEBUG(8, "num solute cg = " << num_solute_cg);

  DEBUG(8, "scaling     : " << t_perturbation_details::do_scaling);
  DEBUG(8, "scaled only : " << sim.param().perturbation.scaled_only);
  
  topology::Chargegroup_Iterator cg1;

  cg1_index = begin;
  cg1_to = num_cg;
  for( ; cg1_index < num_solute_cg; cg1_index+=stride) {

    cg1 = topo.chargegroup_it(cg1_index);

    // intra chargegroup => shortrange
    do_pert_cg_interaction_intra<t_perturbation_details>
      (topo, cg1, pairlist, perturbed_pairlist, sim.param().perturbation.scaled_only);

    // inter chargegroup
    do_pert_cg1_loop(topo, conf, storage, pairlist, perturbed_pairlist,
		     innerloop, perturbed_innerloop,
		     cg1, cg1_index, num_solute_cg, num_cg,
		     periodicity, sim.param().perturbation.scaled_only);
    
  } // cg1

  for( ; cg1_index < num_cg; cg1_index+=stride) {

    // solvent
    cg1 = topo.chargegroup_it(cg1_index);

    // no perturbation here!
    do_cg1_loop(topo, conf, storage, pairlist,
		innerloop,
		cg1, cg1_index, num_solute_cg, num_cg,
		periodicity);
    
  } // cg1

  this->m_timing += util::now() - update_start;
  
  DEBUG(7, "pairlist done");

}

/**
 * loop over chargegroup 1
 */
template<typename t_interaction_spec, typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm
::do_pert_cg1_loop(topology::Topology & topo,
		   configuration::Configuration & conf,
		   interaction::Storage & storage,
		   interaction::Pairlist & pairlist,
		   interaction::Pairlist & perturbed_pairlist,
		   Nonbonded_Innerloop<t_interaction_spec> & innerloop,
		   Perturbed_Nonbonded_Innerloop
		   <t_interaction_spec, t_perturbation_details> & perturbed_innerloop,
		   topology::Chargegroup_Iterator const & cg1,
		   int cg1_index,
		   int const num_solute_cg,
		   int const num_cg,
		   math::Periodicity<t_interaction_spec::boundary_type> const & periodicity,
		   bool scaled_only)
{

  // inter chargegroup
  topology::Chargegroup_Iterator cg2 = *cg1+1;

  // solute...
  int cg2_index;
  math::Vec p;

  for(cg2_index = cg1_index + 1; cg2_index < num_solute_cg; ++cg2, ++cg2_index){
    
    assert(m_cg_cog.size() > cg1_index &&
	   m_cg_cog.size() > cg2_index);
    
    periodicity.nearest_image(m_cg_cog(cg1_index), m_cg_cog(cg2_index), p);
    
    // the distance
    const double d = dot(p, p);
    
    if (d > m_cutoff_long_2){        // OUTSIDE: filter
      continue;
    }
  
    if (d > m_cutoff_short_2){       // LONGRANGE: no filter
      
      topology::Atom_Iterator a1 = cg1.begin(),
	a1_to = cg1.end();
      
      for( ; a1 != a1_to; ++a1){
	for(topology::Atom_Iterator
	      a2 = cg2.begin(),
	      a2_to = cg2.end();
	    a2 != a2_to; ++a2){
	  
	  // the interactions
	  if (topo.is_perturbed(*a1)){

	    if (t_perturbation_details::do_scaling && scaled_only){
	      // ok, only perturbation if it is a scaled pair...
	      std::pair<int, int> 
		energy_group_pair(topo.atom_energy_group(*a1),
				  topo.atom_energy_group(*a2));
	      
	      if (topo.energy_group_scaling().count(energy_group_pair))
		perturbed_innerloop.
		  perturbed_lj_crf_innerloop(topo, conf, *a1, *a2,
					     storage, periodicity);
	      else
		innerloop.
		  lj_crf_innerloop(topo, conf, *a1, *a2,
				   storage, periodicity);
	    } // scaling
	    else{
	      perturbed_innerloop.
		perturbed_lj_crf_innerloop(topo, conf, *a1, *a2,
					   storage, periodicity);
	    }
	  }
	  else if (topo.is_perturbed(*a2)){
	    if (t_perturbation_details::do_scaling && scaled_only){
	      // ok, only perturbation if it is a scaled pair...
	      std::pair<int, int> 
		energy_group_pair(topo.atom_energy_group(*a1),
				  topo.atom_energy_group(*a2));
	      
	      if (topo.energy_group_scaling().count(energy_group_pair))
		perturbed_innerloop.
		  perturbed_lj_crf_innerloop(topo, conf, *a2, *a1,
					     storage, periodicity);
	      else
		
		innerloop.
		  lj_crf_innerloop(topo, conf, *a1, *a2,
				   storage, periodicity);
	    } // scaling
	    else{
	      perturbed_innerloop.
		perturbed_lj_crf_innerloop(topo, conf, *a2, *a1,
					   storage, periodicity);
	    }
	  }
	  else // both unperturbed
	    innerloop.lj_crf_innerloop(topo, conf, *a1, *a2, storage, periodicity);
	} // loop over atom of cg2
      } // loop over atom of cg1

      continue;

    } // longrange

    // SHORTRANGE
    // exclusions! (because cg2 is not solvent)
    do_pert_cg_interaction_excl<t_perturbation_details>
      (topo, cg1, cg2, pairlist, perturbed_pairlist, scaled_only);
    
  } // inter cg (cg2 solute)

  // ? - solvent...
  for(; cg2_index < num_cg; ++cg2, ++cg2_index) {
    
    assert(m_cg_cog.size() > cg1_index &&
	   m_cg_cog.size() > cg2_index);
    
    periodicity.nearest_image(m_cg_cog(cg1_index), m_cg_cog(cg2_index), p);
    
    // the distance
    const double d = dot(p, p);
    
    if (d > m_cutoff_long_2){        // OUTSIDE
      continue;
    }
  
    if (d > m_cutoff_short_2){       // LONGRANGE: calculate interactions!
      
      topology::Atom_Iterator a1 = cg1.begin(),
	a1_to = cg1.end();
      
      for( ; a1 != a1_to; ++a1){
	for(topology::Atom_Iterator
	      a2 = cg2.begin(),
	      a2_to = cg2.end();
	    a2 != a2_to; ++a2){
	  
	  // the interactions
	  if (topo.is_perturbed(*a1)){
	    if (t_perturbation_details::do_scaling && scaled_only){
	      // ok, only perturbation if it is a scaled pair...
	      std::pair<int, int> 
		energy_group_pair(topo.atom_energy_group(*a1),
				  topo.atom_energy_group(*a2));
	      
	      if (topo.energy_group_scaling().count(energy_group_pair))
		perturbed_innerloop.
		  perturbed_lj_crf_innerloop(topo, conf, *a1, *a2,
					     storage, periodicity);
	      else
		innerloop.
		  lj_crf_innerloop(topo, conf, *a1, *a2,
				   storage, periodicity);
	    } // scaling
	    else{
	      perturbed_innerloop.
		perturbed_lj_crf_innerloop(topo, conf, *a1, *a2,
					   storage, periodicity);
	    }
	  }
	  else // second one is solvent, can't be perturbed
	    innerloop.
	      lj_crf_innerloop(topo, conf, *a1, *a2,
			       storage, periodicity, scaled_only);

	} // loop over atom of cg2
      } // loop over atom of cg1

      continue;
    } // longrange

    // SHORTRANGE : at least the second cg is solvent => no exclusions
    do_pert_cg_interaction<t_perturbation_details>
      (topo, cg1, cg2, pairlist, perturbed_pairlist, scaled_only);
    
  } // inter cg (cg2 solvent)
}

/**
 * inter cg, no exclusion
 */
template<typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm
::do_pert_cg_interaction(topology::Topology & topo,
			 topology::Chargegroup_Iterator const &cg1,
			 topology::Chargegroup_Iterator const &cg2,
			 interaction::Pairlist & pairlist,
			 interaction::Pairlist & perturbed_pairlist,
			 bool scaled_only)
{

  topology::Atom_Iterator a1 = cg1.begin(),
    a1_to = cg1.end();

  DEBUG(11, "do_cg_interaction " << *a1);
    
  for( ; a1 != a1_to; ++a1){
    for(topology::Atom_Iterator
	  a2 = cg2.begin(),
	  a2_to = cg2.end();
	a2 != a2_to; ++a2){

      if (topo.is_perturbed(*a1)){

	if (t_perturbation_details::do_scaling && scaled_only){
	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(*a1),
			      topo.atom_energy_group(*a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[*a1].push_back(*a2);
	  else
	    pairlist[*a1].push_back(*a2);
	} // scaling
	else{
	  perturbed_pairlist[*a1].push_back(*a2);
	}
      }
      else if (topo.is_perturbed(*a2)){
	if (t_perturbation_details::do_scaling && scaled_only){
	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(*a1),
			      topo.atom_energy_group(*a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[*a2].push_back(*a1);
	  else
	    pairlist[*a1].push_back(*a2);
	} // scaling
	else{
	  perturbed_pairlist[*a2].push_back(*a1);
	}
      }
      else
	pairlist[*a1].push_back(*a2);

    } // loop over atom 2 of cg1
  } // loop over atom 1 of cg1
}

template<typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm
::do_pert_cg_interaction_excl(topology::Topology & topo,
			      topology::Chargegroup_Iterator const & cg1,
			      topology::Chargegroup_Iterator const & cg2,
			      interaction::Pairlist & pairlist,
			      interaction::Pairlist & perturbed_pairlist,
			      bool scaled_only)
{
  topology::Atom_Iterator a1 = cg1.begin(),
    a1_to = cg1.end();

  DEBUG(11, "do_cg_interaction_excl " << *a1);
  
  for( ; a1 != a1_to; ++a1){
    for(topology::Atom_Iterator
	  a2 = cg2.begin(),
	  a2_to = cg2.end();
	a2 != a2_to; ++a2){

      // check it is not excluded
      if (excluded_solute_pair(topo, *a1, *a2))
	continue;

      if (topo.is_perturbed(*a1)){
	if (t_perturbation_details::do_scaling && scaled_only){
	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(*a1),
			      topo.atom_energy_group(*a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[*a1].push_back(*a2);
	  else
	    pairlist[*a1].push_back(*a2);
	} // scaling
	else{
	  perturbed_pairlist[*a1].push_back(*a2);
	}
      }
      else if (topo.is_perturbed(*a2)){
	if (t_perturbation_details::do_scaling && scaled_only){
	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(*a1),
			      topo.atom_energy_group(*a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[*a2].push_back(*a1);
	  else
	    pairlist[*a1].push_back(*a2);
	} // scaling
	else{
	  perturbed_pairlist[*a2].push_back(*a1);
	}
      }
      else // both unperturbed
	pairlist[*a1].push_back(*a2);

    } // loop over atom 2 of cg1
  } // loop over atom 1 of cg1
}

template<typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm
::do_pert_cg_interaction_intra(topology::Topology & topo,
			       topology::Chargegroup_Iterator const & cg1,
			       interaction::Pairlist & pairlist,
			       interaction::Pairlist & perturbed_pairlist,
			       bool scaled_only)
{
  topology::Atom_Iterator a1 = cg1.begin(),
    a1_to = cg1.end();

  DEBUG(11, "do_cg_interaction_intra " << *a1);
  
  for( ; a1 != a1_to; ++a1){
    for(topology::Atom_Iterator
	  a2(*a1+1);
	a2 != a1_to; ++a2){

      // check it is not excluded
      if (excluded_solute_pair(topo, *a1, *a2))
	continue;

      if (topo.is_perturbed(*a1)){
	if (t_perturbation_details::do_scaling && scaled_only){
	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(*a1),
			      topo.atom_energy_group(*a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[*a1].push_back(*a2);
	  else
	    pairlist[*a1].push_back(*a2);
	} // scaling
	else{
	  perturbed_pairlist[*a1].push_back(*a2);
	}
      }
      else if (topo.is_perturbed(*a2)){
	if (t_perturbation_details::do_scaling && scaled_only){
	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(*a1),
			      topo.atom_energy_group(*a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[*a2].push_back(*a1);
	  else
	    pairlist[*a1].push_back(*a2);
	} // scaling
	else{
	  perturbed_pairlist[*a2].push_back(*a1);
	}
      }
      else
	pairlist[*a1].push_back(*a2);
      
    } // loop over atom 2 of cg1
  } // loop over atom 1 of cg1
}


////////////////////////////////////////////////////////////////////////////////
// atomic cutoff
////////////////////////////////////////////////////////////////////////////////

template<typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm::
update_pert_atomic(topology::Topology & topo,
		   configuration::Configuration & conf,
		   simulation::Simulation & sim, 
		   interaction::Storage & storage,
		   interaction::Pairlist & pairlist,
		   interaction::Pairlist & perturbed_pairlist,
		   unsigned int begin, unsigned int end,
		   unsigned int stride)
{
  SPLIT_PERT_INNERLOOP(_update_pert_atomic,
		       topo, conf, sim, storage,
		       pairlist, perturbed_pairlist, 
		       begin, end, stride);
}

template<typename t_interaction_spec, typename t_perturbation_details>
void interaction::Standard_Pairlist_Algorithm::
_update_pert_atomic(topology::Topology & topo,
		    configuration::Configuration & conf,
		    simulation::Simulation & sim,
		    interaction::Storage & storage,
		    interaction::Pairlist & pairlist,
		    interaction::Pairlist & perturbed_pairlist,
		    unsigned int begin, unsigned int end,
		    unsigned int stride)
{
  DEBUG(7, "standard pairlist update (atomic cutoff)");
  const double update_start = util::now();
  
  // create the innerloops
  math::Periodicity<t_interaction_spec::boundary_type> periodicity(conf.current().box);
  
  Nonbonded_Innerloop<t_interaction_spec> innerloop(*m_param);
  innerloop.init(sim);
  
  Perturbed_Nonbonded_Innerloop<t_interaction_spec, t_perturbation_details>
    perturbed_innerloop(*m_param);
  perturbed_innerloop.init(sim);
  perturbed_innerloop.set_lambda(topo.lambda(), topo.lambda_exp());

  // empty the pairlist
  assert(pairlist.size() == topo.num_atoms());
  assert(perturbed_pairlist.size() == topo.num_atoms());
  
  for(unsigned int i=0; i<topo.num_atoms(); ++i){
    pairlist[i].clear();
    perturbed_pairlist[i].clear();
  }

  DEBUG(7, "pairlist cleared");

  const int num_solute = topo.num_solute_atoms();
  const int num_atoms = topo.num_atoms();

  math::VArray const & pos = conf.current().pos;
  math::Vec v;
  
  for(int a1 = 0; a1 < num_solute; a1 += stride) {
    DEBUG(9, "solute (" << a1 << ") - solute");

    for(int a2 = a1 + 1; a2 < num_solute; ++a2){

      // check exclusions and range
      periodicity.nearest_image(pos(a1), pos(a2), v);
      
      // the distance
      const double d = dot(v, v);

      DEBUG(10, "\t" << a1 << " - " << a2);
    
      if (d > m_cutoff_long_2){        // OUTSIDE
	DEBUG(11, "\t\toutside");
	continue;
      }
  
      if (d > m_cutoff_short_2){       // LONGRANGE: calculate interactions!
	DEBUG(11, "\t\tlongrange");
	
	// the interactions
	if (topo.is_perturbed(a1)){
	  DEBUG(11, "\t\t" << a1 << " perturbed");

	  if (t_perturbation_details::do_scaling &&
	      sim.param().perturbation.scaled_only){

	    // ok, only perturbation if it is a scaled pair...
	    std::pair<int, int> 
	      energy_group_pair(topo.atom_energy_group(a1),
				topo.atom_energy_group(a2));
	    
	    if (topo.energy_group_scaling().count(energy_group_pair))
	      perturbed_innerloop.
		perturbed_lj_crf_innerloop(topo, conf, a1, a2,
					   storage, periodicity);
	    else
	      innerloop.
		lj_crf_innerloop(topo, conf, a1, a2,
				 storage, periodicity);
	  } // perturb scaled interactions only
	  else{
	    perturbed_innerloop.
	      perturbed_lj_crf_innerloop(topo, conf, a1, a2,
					 storage, periodicity);
	  }
	}
	else if (topo.is_perturbed(a2)){
	  DEBUG(11, "\t\t" << a2 << " perturbed");

	  if (t_perturbation_details::do_scaling &&
	      sim.param().perturbation.scaled_only){

	    // ok, only perturbation if it is a scaled pair...
	    std::pair<int, int> 
	      energy_group_pair(topo.atom_energy_group(a1),
				topo.atom_energy_group(a2));
	    
	    if (topo.energy_group_scaling().count(energy_group_pair))
	      perturbed_innerloop.
		perturbed_lj_crf_innerloop(topo, conf, a2, a1,
					   storage, periodicity);
	    else
	      innerloop.
		lj_crf_innerloop(topo, conf, a1, a2,
				 storage, periodicity);
	  } // perturb scaled interactions only
	  else{
	    perturbed_innerloop.
	      perturbed_lj_crf_innerloop(topo, conf, a2, a1,
					 storage, periodicity);
	  }
	}
	else{
	  DEBUG(11, "\t\tnot perturbed");
	  innerloop.lj_crf_innerloop(topo, conf, a1, a2, storage, periodicity);
	}
	
	continue;
      } // longrange
      
      // shortrange - check exclusions
      if (excluded_solute_pair(topo, a1, a2)) continue;

      DEBUG(11, "\t\tshortrange");

      if (topo.is_perturbed(a1)){
	DEBUG(11, "\t\t" << a1 << " perturbed");

	if (t_perturbation_details::do_scaling &&
	    sim.param().perturbation.scaled_only){

	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(a1),
			      topo.atom_energy_group(a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[a1].push_back(a2);
	  else
	    pairlist[a1].push_back(a2);
	} // scaling
	else{
	  perturbed_pairlist[a1].push_back(a2);
	}
      }
      else if (topo.is_perturbed(a2)){
	DEBUG(11, "\t\t" << a2 << " perturbed");

	if (t_perturbation_details::do_scaling &&
	    sim.param().perturbation.scaled_only){

	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(a1),
			      topo.atom_energy_group(a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[a2].push_back(a1);
	  else
	    pairlist[a1].push_back(a2);
	} // scaling
	else{
	  perturbed_pairlist[a2].push_back(a1);
	}
	
      }
      else{
	DEBUG(11, "\t\tnot perturbed");
	pairlist[a1].push_back(a2);
      }
      
    } // solute - solute

    DEBUG(9, "solute (" << a1 << ") - solvent");

    for(int a2 = num_solute; a2 < num_atoms; ++a2){
      
      // check range, no exclusions
      periodicity.nearest_image(pos(a1), pos(a2), v);
      
      // the distance
      const double d = dot(v, v);
    
      DEBUG(10, "\t" << a1 << " - " << a2);

      if (d > m_cutoff_long_2){        // OUTSIDE
	DEBUG(11, "\t\toutside");
	continue;
      }
  
      if (d > m_cutoff_short_2){       // LONGRANGE: calculate interactions!
	DEBUG(11, "\t\tlongrange");
	
	// the interactions
	if (topo.is_perturbed(a1)){

	  DEBUG(11, "\t\t" << a1 << " perturbed");
	  if (t_perturbation_details::do_scaling &&
	      sim.param().perturbation.scaled_only){

	    // ok, only perturbation if it is a scaled pair...
	    std::pair<int, int> 
	      energy_group_pair(topo.atom_energy_group(a1),
				topo.atom_energy_group(a2));
	    
	    if (topo.energy_group_scaling().count(energy_group_pair))
	      perturbed_innerloop.
		perturbed_lj_crf_innerloop(topo, conf, a1, a2, storage, periodicity);
	    else
	      innerloop.
		lj_crf_innerloop(topo, conf, a1, a2, storage, periodicity);

	  } // scaling
	  else{
	    perturbed_innerloop.
	      perturbed_lj_crf_innerloop(topo, conf, a1, a2, storage, periodicity);
	  }
	}
	else{
	  DEBUG(11, "\t\tnot perturbed");
	  innerloop.lj_crf_innerloop(topo, conf, a1, a2, storage, periodicity);
	}
	
	continue;
      } // longrange
      
      DEBUG(11, "\t\tshortrange");

      if (topo.is_perturbed(a1)){
	DEBUG(11, "\t\t" << a1 << " perturbed");

	if (t_perturbation_details::do_scaling &&
	    sim.param().perturbation.scaled_only){

	  // ok, only perturbation if it is a scaled pair...
	  std::pair<int, int> 
	    energy_group_pair(topo.atom_energy_group(a1),
			      topo.atom_energy_group(a2));
	  
	  if (topo.energy_group_scaling().count(energy_group_pair))
	    perturbed_pairlist[a1].push_back(a2);
	  else
	    pairlist[a1].push_back(a2);

	} // scaling
	else{
	  perturbed_pairlist[a1].push_back(a2);
	}
      }
      else{
	DEBUG(11, "\t\tnot perturbed");
	pairlist[a1].push_back(a2);
      }

    } // solute - solvent
    
  }

  int a1 = num_solute;

  // multiple solvents
  DEBUG(9, "solvent - solvent");
  DEBUG(10, "\tnum_atoms = " << num_atoms);

  for(unsigned int s = 0; s < topo.num_solvents(); ++s){
    DEBUG(11, "solvent " << s);

    int end = a1 + topo.num_solvent_atoms(s);
    DEBUG(11, "\tends at atom " << end);

    const int num_solv_at = topo.num_solvent_atoms(s) / topo.num_solvent_molecules(s);
    int a2_start = a1 + num_solv_at;
    DEBUG(11, "\twith " << num_solv_at << " atoms");
    
    for( ; a1 < end; ++a1){
      
      if (a1 == a2_start) a2_start += num_solv_at;
      
      for(int a2 = a2_start; a2 < num_atoms; ++a2){
	
	// check range, no exclusions
	periodicity.nearest_image(pos(a1), pos(a2), v);
	
	// the distance
	const double d = dot(v, v);

	DEBUG(10, "\t" << a1 << " - " << a2);
	
	if (d > m_cutoff_long_2){        // OUTSIDE
	  DEBUG(11, "\t\toutside");
	  continue;
	}
  
	if (d > m_cutoff_short_2){       // LONGRANGE: calculate interactions!
	  DEBUG(11, "\t\tlongrange");	  

	  // the interactions
	  innerloop.lj_crf_innerloop(topo, conf, a1, a2, storage, periodicity);
	  
	  continue;
	} // longrange

	DEBUG(11, "\t\tshortrange");	
	pairlist[a1].push_back(a2);
	
      } // solvent - solvent

    } // a1 of solvent s

  } // multiple solvents
  

  this->m_timing += util::now() - update_start;
  
  DEBUG(7, "pairlist done");

}


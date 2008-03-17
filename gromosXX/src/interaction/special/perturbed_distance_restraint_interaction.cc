/**
 * @file perturbed_distance_restraint_interaction.cc
 * template methods of Perturbed_Distance_Restraint_Interaction
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>

#include <math/periodicity.h>

// special interactions
#include <interaction/interaction_types.h>

#include <interaction/special/perturbed_distance_restraint_interaction.h>

#include <util/template_split.h>
#include <util/debug.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

/**
 * calculate position restraint interactions
 */
template<math::boundary_enum B, math::virial_enum V>
static int _calculate_perturbed_distance_restraint_interactions
(topology::Topology & topo,
 configuration::Configuration & conf,
 simulation::Simulation & sim,
 double exponential_term)
{
  // loop over the perturbed distance restraints
  std::vector<topology::perturbed_distance_restraint_struct>::const_iterator 
    it = topo.perturbed_distance_restraints().begin(),
    to = topo.perturbed_distance_restraints().end();
  
  std::vector<double>::iterator ave_it = conf.special().distrest_av.begin();

  // math::VArray &pos   = conf.current().pos;
  // math::VArray &force = conf.current().force;
  math::Vec v, f;

  double en_term, dlam_term, energy, energy_derivativ;


  math::Periodicity<B> periodicity(conf.current().box);

  DEBUG(7, "perturbed distance restraint interactions : " 
	<< topo.perturbed_distance_restraints().size());

  for( ; it != to; ++it){

    periodicity.nearest_image(it->v1.pos(conf), it->v2.pos(conf), v);
#ifndef NDEBUG
    for(int i=0;i<it->v1.size();i++){
      DEBUG(10, "v1 (atom " << i+1 << "): " << it->v1.atom(i)+1);
    }
    for(int i=0;i<it->v2.size();i++){
      DEBUG(10, "v2 (atom " << i+1 << "): " << it->v2.atom(i)+1);
    }
#endif 
  
    DEBUG(9, "PERTDISTREST v : " << math::v2s(v));
   
    double dist = abs(v);
   
    const double l = topo.lambda();
    const double w0 = (1-l)*it->A_w0 + l*it->B_w0;
    const double r0 = (1-l)*it->A_r0 + l*it->B_r0;
    const double K = sim.param().distrest.K;
    const double r_l = sim.param().distrest.r_linear; 
    const double D_r0 = it->B_r0 - it->A_r0;
    
    DEBUG(9, "PERTDISTREST dist : " << dist << " r0 " << r0 << " rah " << it->rah);  
    if (sim.param().distrest.distrest < 0) {
      (*ave_it) = (1.0 - exponential_term) * pow(dist, -3.0) + 
                   exponential_term * (*ave_it);
      dist = pow(*ave_it, -1.0 / 3.0);
      DEBUG(9, "PERTDISTREST average dist : " << dist);
    }

    double prefactor = pow(2, it->n + it->m) * pow(l, it->n) * pow(1-l, it->m);
    
    if(it->rah*dist < it->rah*(r0)){
	
      DEBUG(9, "PERTDISTREST  : (rep / attr) restraint fulfilled");
      f=0*v;
      en_term = 0;
    }    
    else if(fabs(r0 - dist) < r_l){

      DEBUG(9, "PERTDISTREST  :  harmonic");
      f = -K*(dist - (r0))*v / dist; 
      en_term = 0.5 * K * (dist - r0) * (dist - r0);
    }    
    else{
      DEBUG(9, "PERTDISTREST  : (rep / attr) linear");
      if(dist<r0){
 	DEBUG(9, "PERTDISTREST   : (rep / attr) linear negative");
	f = r_l * K * v / dist;
	en_term = -K * (dist + 0.5 * r_l - r0) * r_l;
      }
      else{
 	DEBUG(9, "PERTDISTREST   : (rep / attr) linear positive");
	f = -r_l * K * v / dist;
	en_term = K * (dist - 0.5 * r_l - r0) * r_l;
      }
    }
  
    if(abs(sim.param().distrest.distrest) == 1)
      ;      
    else if(abs(sim.param().distrest.distrest) == 2){
      f=f*w0;
      en_term = en_term*w0;
    }
    else{
      f=f*0;
      en_term=0;
    }
    f = prefactor*f;
    energy = prefactor*en_term;
    
    it->v1.force(conf,  f);
    it->v2.force(conf, -f);  
    
    DEBUG(10, "PERTDISTREST Energy : " << energy);
    conf.current().energies.distrest_energy[topo.atom_energy_group()
					    [it->v1.atom(0)]] += energy;
    
    if(it->rah*dist <it->rah*(r0))
      dlam_term = 0;
    
    else if(abs(sim.param().distrest.distrest) == 1){
      if(fabs(r0-dist)<r_l){
	// harmonic
	dlam_term = -K * ( dist - r0 ) * D_r0;
      }
      else {
	if(dist<r0){
	  // linear negative
	  dlam_term = K * D_r0 * r_l;
	}
	else{
	  // linear positive
	  dlam_term = -K * D_r0 * r_l;
	}
	
      }	
    }
    else if(abs(sim.param().distrest.distrest) == 2){
      if(fabs(r0-dist)<r_l){
	// harmonic
	dlam_term = 0.5 * K * (it->B_w0 - it->A_w0) * (dist - r0)*(dist - r0)
	  - K * w0 * (dist - r0) * D_r0;
      }
      else {
	if(dist<r0){
	  // linear negative
	  dlam_term = -K *(it->B_w0 - it->A_w0) * (dist + 0.5*r_l - r0) * r_l
	    + K * w0 * D_r0 * r_l;
	}
	else{   
	  // linear positive
	  dlam_term =  K *(it->B_w0 - it->A_w0) * (dist - 0.5 * r_l - r0) * r_l
	    - K * w0 * D_r0 * r_l;
	}
      }

    }

    // the derivative of the prefactor
    // division by zero precaution
    double dprefndl, dprefmdl;
    if (it->n==0) dprefndl = 0;
    else dprefndl = it->n * pow(l, it->n-1) * pow(1 - l, it->m);
    
    if (it->m == 0) dprefmdl = 0;
    else dprefmdl = it->m * pow(l, it->n) * pow(1 - l, it->m-1);

    double dprefdl = pow(2, it->m + it->n) * 
      (dprefndl - dprefmdl) * en_term;
    
    double dpotdl = prefactor * dlam_term;

    energy_derivativ = dprefdl + dpotdl;
    DEBUG(10, "PERTDISTREST Energy derivative : " << energy_derivativ);

    conf.current().perturbed_energy_derivatives.distrest_energy[topo.atom_energy_group()
								[it->v1.atom(0)]] += energy_derivativ;
    
  }

  return 0;
}

/**
 * calculate position restraint interactions
 */
template<math::boundary_enum B>
static void _init_averages
(topology::Topology & topo,
 configuration::Configuration & conf)
{
  math::Periodicity<B> periodicity(conf.current().box);
  math::Vec v;
  
  for(std::vector<topology::perturbed_distance_restraint_struct>::const_iterator
        it = topo.perturbed_distance_restraints().begin(),
        to = topo.perturbed_distance_restraints().end(); it != to; ++it) {
    periodicity.nearest_image(it->v1.pos(conf), it->v2.pos(conf), v);
    conf.special().distrest_av.push_back(pow(math::abs(v), -3.0));
  }
}

int interaction::Perturbed_Distance_Restraint_Interaction
::calculate_interactions(topology::Topology &topo,
			 configuration::Configuration &conf,
			 simulation::Simulation &sim)
{
  SPLIT_VIRIAL_BOUNDARY(_calculate_perturbed_distance_restraint_interactions,
			topo, conf, sim, exponential_term);
  
  return 0;
}

int interaction::Perturbed_Distance_Restraint_Interaction
::init(topology::Topology &topo, 
       configuration::Configuration &conf,
       simulation::Simulation &sim,
       std::ostream &os,
       bool quiet) 
{
  if (sim.param().distrest.distrest < 0) {
    exponential_term = std::exp(- sim.time_step_size() / 
                                  sim.param().distrest.tau);
    
    if (!sim.param().distrest.read) {
      // reset averages to r_0
      SPLIT_BOUNDARY(_init_averages, topo, conf);
    }
  }
  
  if (!quiet) {
    os << "Perturbed distance restraint interaction";
    if (sim.param().distrest.distrest < 0)
      os << "with time-averaging";
    os << std::endl;
  }
  return 0;
}

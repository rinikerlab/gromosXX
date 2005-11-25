/**
 * @file jvalue_restraint_interaction.cc
 * template methods of Jvalue_Restraint_Interaction
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <math/periodicity.h>

// special interactions
#include <interaction/interaction_types.h>

#include <interaction/special/jvalue_restraint_interaction.h>

#include "create_special.h"

#include <util/template_split.h>
#include <util/debug.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

double _calculate_derivative(topology::Topology & topo,
			     configuration::Configuration &conf,
			     simulation::Parameter const & param,
			     std::vector<topology::jvalue_restraint_struct>::const_iterator it,
			     double phi, double Jcurr, double Jav,
			     double cos_phi_delta, double sin_phi_delta);


/**
 * calculate jvalue restraint interactions
 */
template<math::boundary_enum B, math::virial_enum V>
int _calculate_jvalue_restraint_interactions
(topology::Topology & topo,
 configuration::Configuration & conf,
 simulation::Simulation & sim)
{
  std::cout.precision(5);
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  
  // loop over the jvalue restraints
  std::vector<topology::jvalue_restraint_struct>::iterator 
    it = topo.jvalue_restraints().begin(),
    to = topo.jvalue_restraints().end();

  math::VArray & pos = conf.current().pos;
  math::Vec rij, rkj, rkl, rmj, rnk, rim, rln;

  double dkj2, dmj2, dnk2, dim, dln;

  math::Periodicity<B> periodicity(conf.current().box);

  int n = 0;
  for( ; it != to; ++it, ++n){

    //get nearest image, calculate rij, rkj, rkl
    periodicity.nearest_image(pos(it->i), pos(it->j), rij);
    periodicity.nearest_image(pos(it->k), pos(it->j), rkj);
    periodicity.nearest_image(pos(it->k), pos(it->l), rkl);
    
    //calculate phi, cross- and dot-products
    rmj = cross(rij, rkj);
    rnk = cross(rkj, rkl);
    dkj2 = abs2(rkj);
    dmj2 = abs2(rmj);
    dnk2 = abs2(rnk);
    
    const double frim = dot(rij, rkj) / dkj2;
    const double frln = dot(rkl, rkj) / dkj2;

    rim = rij - frim * rkj;
    rln = frln * rkj - rkl;
    dim = sqrt(abs2(rim));
    dln = sqrt(abs2(rln));
    
    const double ip = dot(rim, rln);
    const double cosphi = ip / (dim*dln);
    
    // double phi = acos(dot(rmj, rnk)/(sqrt(dmj2)*sqrt(dnk2)));
    double phi = acos(cosphi);
    
    const double sign = dot(rij, rnk);

    if(sign < 0){ 
      phi = -phi;
    }
    
    // check this!!!!
    DEBUG(10, "JVAL phi: " << 180 * phi / math::Pi);
    
    const double cos_phi_delta = cos(phi + it->delta);
    const double sin_phi_delta = sin(phi + it->delta);

    ////////////////////////////////////////////////////////////////////////////////

    double exp_term;
    double memory_decay;

    //decide on time averaging
    if (sim.param().jvalue.mode != simulation::restr_inst){
 
      // time averaging 
      exp_term = exp(-sim.time_step_size() / sim.param().jvalue.tau);
      memory_decay = 1 - exp(-sim.time_step_size() / sim.param().jvalue.tau);
    }
    else{
      
      // instantaneous J-values
      exp_term = 0;
      memory_decay = 1;
    }

    DEBUG(10, "JVAL exp_term " << exp_term);
    DEBUG(10, "JVAL memory_decay " << memory_decay);
        
    // calculate J-value
    assert(conf.special().jvalue_av.size() > unsigned(n));
    
    const double Jcurr =
      it->a * cos_phi_delta * cos_phi_delta +
      it->b * cos_phi_delta +
      it->c;

    const double Jav =
      memory_decay * Jcurr +
      conf.special().jvalue_av[n] * exp_term;

    DEBUG(8, "JDATA time: " << sim.time() << "\t\tJcurr: " << Jcurr
	  << "\tJav: " << Jav << "\tJ0: " << it->J0);
    
    //write new average and current value
    conf.special().jvalue_av[n] = Jav;
    conf.special().jvalue_curr[n] = Jcurr;
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // LOCAL ELEVATION RESTRAINING
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    if (sim.param().jvalue.le){
      while(phi < 0.0)
	phi += 2 * math::Pi;
      while(phi >= 2 * math::Pi)
	phi -= 2 * math::Pi;
      
      // 0 .. ngrid-1
      const int bin = int(phi * sim.param().jvalue.ngrid / (2 * math::Pi));
      DEBUG(8, "jelevation: phi=" << 180 * phi / math::Pi 
	    << " phi0=" << (bin + 0.5) * 360 / sim.param().jvalue.ngrid << " bin=" << bin);
      
      assert(it->epsilon.size() > unsigned(bin));
      double delta_epsilon;
      double delta_Jav = Jav - it->J0;
      double delta_Jinst = Jcurr - it->J0;

      // zero potential energy within J0 +- delta
      // (no elevation if J is ~ correct)
      if (delta_Jav > 0){
	if (delta_Jav > sim.param().jvalue.delta)
	  delta_Jav -= sim.param().jvalue.delta;
	else delta_Jav = 0.0;
      }
      else{
	if (delta_Jav < sim.param().jvalue.delta)
	  delta_Jav += sim.param().jvalue.delta;
	else delta_Jav = 0.0;
      }

      if (delta_Jinst > 0){
	if (delta_Jinst > sim.param().jvalue.delta)
	  delta_Jinst -= sim.param().jvalue.delta;
	else delta_Jinst = 0.0;
      }
      else{
	if (delta_Jinst < sim.param().jvalue.delta)
	  delta_Jinst += sim.param().jvalue.delta;
	else delta_Jinst = 0.0;
      }
      
      if (sim.param().jvalue.mode == simulation::restr_biq)
	delta_epsilon = delta_Jinst*delta_Jinst * delta_Jav * delta_Jav;
      else if (sim.param().jvalue.mode == simulation::restr_inst)
	delta_epsilon = delta_Jinst * delta_Jinst;	
      else if (sim.param().jvalue.mode == simulation::restr_av)
	delta_epsilon = delta_Jav * delta_Jav;

      it->epsilon[bin] += delta_epsilon;

      DEBUG(8, "jelevation: epsilon += " << delta_epsilon);

    }

    const double dV_dphi = 
      _calculate_derivative(topo, conf, sim.param(),
			    it,
			    phi, Jcurr, Jav,
			    cos_phi_delta, sin_phi_delta);
    
    DEBUG(10, "derivative calculated");

    //calculate forces 		 
    const math::Vec dphi_dri =  (sqrt(dkj2)/dmj2)*rmj;
    const math::Vec dphi_drl = -(sqrt(dkj2)/dnk2)*rnk;			
    const math::Vec dphi_drj = (frim -1)*dphi_dri - frln*dphi_drl;
    const math::Vec dphi_drk = -1.0*dphi_dri - dphi_drj - dphi_drl;			
 
    const math::Vec fi = - dV_dphi * dphi_dri;
    const math::Vec fj = - dV_dphi * dphi_drj;
    const math::Vec fk = - dV_dphi * dphi_drk; 
    const math::Vec fl = - dV_dphi * dphi_drl;
    
    DEBUG(10, "JVAL Force on i " << math::v2s(fi) );
    DEBUG(10, "JVAL Force on j " << math::v2s(fj) );
    DEBUG(10, "JVAL Force on k " << math::v2s(fk) );
    DEBUG(10, "JVAL Force on l " << math::v2s(fl) );
   
    conf.current().force(it->i) += fi;
    conf.current().force(it->j) += fj;
    conf.current().force(it->k) += fk;
    conf.current().force(it->l) += fl;

    if (V == math::atomic_virial){
      math::Vec rlj;
      periodicity.nearest_image(pos(it->l), pos(it->j), rlj);
      
      for(int d0=0; d0<3; ++d0)
	for(int d1=0; d1<3; ++d1)
	  conf.current().virial_tensor(d0, d1) += 
	    rij(d0) * fi(d1) +
	    rkj(d0) * fk(d1) +
	    rlj(d0) * fl(d1);
      
      DEBUG(7, "\tatomic virial done");
    }
  }

  return 0;
}

int interaction::Jvalue_Restraint_Interaction::calculate_interactions
(
 topology::Topology & topo,
 configuration::Configuration & conf,
 simulation::Simulation & sim)
{
  SPLIT_VIRIAL_BOUNDARY(_calculate_jvalue_restraint_interactions,
			topo, conf, sim);
  return 0;
}

/**
 * init
 */
int interaction::Jvalue_Restraint_Interaction::init
(
 topology::Topology &topo, 
 configuration::Configuration &conf,
 simulation::Simulation &sim,
 std::ostream &os,
 bool quiet) 
{
  if (!quiet){
    os << "J-value restraint interaction\n";
    switch(sim.param().jvalue.mode){
      case simulation::restr_off:
	os << "\trestraining off\n";
	break;
      case simulation::restr_inst:
	os << "\tinstantaneous restraining\n";
	break;
      case simulation::restr_av:
	os << "\ttime averaged restraining\n";
	break;
      case simulation::restr_biq:
	os << "\tbiquadratic restraining\n";
	break;
    }

    if (sim.param().jvalue.read_av)
      os << "\treading jvalue averages from file\n";
  }
  
  // loop over the jvalue restraints
  std::vector<topology::jvalue_restraint_struct>::iterator 
    it = topo.jvalue_restraints().begin(),
    to = topo.jvalue_restraints().end();

  if (sim.param().jvalue.le){
    if (!quiet)
      os << "\tlocal elevation restraining enabled\n";
    
    if (!sim.param().jvalue.read_av){
      for( ; it != to; ++it){
	it->epsilon.resize(sim.param().jvalue.ngrid, 0.0);
      }
    } 
    else
      os << "\treading local elevation epsilon from file\n";
  }
  
  return 0;
};

double _calculate_derivative(topology::Topology & topo,
			     configuration::Configuration &conf,
			     simulation::Parameter const & param,
			     std::vector<topology::jvalue_restraint_struct>::const_iterator it,
			     double phi, double Jcurr, double Jav,
			     double cos_phi_delta, double sin_phi_delta)
{
  if (param.jvalue.le){
    
    // LOCAL ELEVATION RESTRAINING

    double energy = 0.0, dV_dphi = 0.0;
    const double K = it->K * param.jvalue.K;
    
    for(int i=0; i < param.jvalue.ngrid; ++i){
      
      DEBUG(10, "le potential " << i);
      
      const double phi0 = (i + 0.5) * 2 * math::Pi / param.jvalue.ngrid;
      const double w = 2 * math::Pi / param.jvalue.ngrid;

      // adjust periodicity
      while(phi < phi0 - math::Pi)
	phi += 2 * math::Pi;
      while(phi >= phi0 + math::Pi)
	phi -= 2 * math::Pi;
      
      const double delta_phi = phi - phi0;
      DEBUG(10, "\tdelta_phi = " << delta_phi);
      
      const double Vpen = it->epsilon[i] * K *
	exp(- delta_phi * delta_phi / (2 * w * w));
      
      DEBUG(10, "\tenergy = " << Vpen);

      energy += Vpen;
      dV_dphi -= Vpen * delta_phi / (w * w);

    }

    conf.current().energies.jvalue_energy[topo.atom_energy_group()[it->i]]
      += energy;
    return dV_dphi;
        
  }
  else{

    // STANDARD RESTRAINIG

    // instantaneous / time averaged
    if (param.jvalue.mode == simulation::restr_inst ||
	param.jvalue.mode == simulation::restr_av){
      
      // check for half - harmoic functional forms
      if ( (it->H == topology::repulsive && Jav - it->J0 > 0) || 
	 (it->H == topology::attractive && Jav - it->J0 <= 0) ){
	return 0;
      }
      else{
	const double K = it->K * param.jvalue.K;
	
	// calculate derivatives + energy	
	// Jav == Jcurr for instantaneous...
	const double dV_dJ = K * (Jav - it->J0);
	
	// memory_decay factor is omitted for practical reasons.
	const double dJ_dphi = - (2 * it->a * cos_phi_delta * sin_phi_delta + it->b * sin_phi_delta);
	
	const double energy = 0.5 * K * (Jav - it->J0) * (Jav - it->J0);
	// and store...
	conf.current().energies.jvalue_energy[topo.atom_energy_group()[it->i]]
	  += energy;
	
	return dV_dJ * dJ_dphi;
      }
    }
    else if (param.jvalue.mode == simulation::restr_biq){
      
      // check for half - harmoic functional forms
      if ( (it->H == topology::repulsive && (Jcurr - it->J0 > 0 || Jav - it->J0 > 0)) || 
	   (it->H == topology::attractive && (Jcurr - it->J0 <= 0 || Jav - it->J0 <= 0)) ){
	return 0;
      }
      else{
	// calculate derivatives + energy	
	const double K = it->K * param.jvalue.K;
	const double delta_Jcurr = Jcurr - it->J0;
	const double delta_Jav = Jav - it->J0;
	
	// --- dV / dJ ---
	const double dV_dJ = K * delta_Jcurr * delta_Jav * delta_Jav;
	// memory_decay factor is omitted for practical reasons.
	const double dJ_dphi = - (2 * it->a * cos_phi_delta * sin_phi_delta + it->b * sin_phi_delta);
	
	// --- dV / dJav ---
	const double dV_dJav = K * delta_Jcurr * delta_Jcurr * delta_Jav;
	// memory_decay factor is omitted for practical reasons.
	const double dJav_dphi = - (2 * it->a * cos_phi_delta * sin_phi_delta + it->b * sin_phi_delta);
	
	const double energy = 0.5 * K
	  * delta_Jcurr * delta_Jcurr * delta_Jav * delta_Jav;
	
	conf.current().energies.jvalue_energy[topo.atom_energy_group()[it->i]]
	  += energy;
	
	return dV_dJ * dJ_dphi + dV_dJav * dJav_dphi;
      }
    }
    
    io::messages.add("JValue restrints: derivative not implemented!",
		     "JValue_Restraint_Interaction",
		     io::message::critical);
  }

  return 0;
  
}

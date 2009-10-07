/**
 * @file crossdihedral_interaction.cc
 * template methods of Crossdihedral_Interaction.
 * calculates the crossdihedral forces for any m and any shift angle
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>

#include <math/periodicity.h>

// interactions
#include <interaction/interaction_types.h>
#include "crossdihedral_interaction.h"

#include <util/template_split.h>
#include <util/debug.h>
#include <util/error.h>

#include <iomanip>
#include <iostream>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE bonded

#include <util/debug.h>


static double _calculate_nearest_minimum(double phi, int m, double cospd);

/**
 * calculate crossdihedral forces and energies.
 */
template<math::boundary_enum B, math::virial_enum V>
static int _calculate_crossdihedral_interactions(topology::Topology & topo,
					    configuration::Configuration & conf,
					    simulation::Simulation & sim,
					    std::vector<interaction::dihedral_type_struct> 
					    const & param)
{
  // loop over the crossdihedrals
  std::vector<topology::eight_body_term_struct>::iterator d_it =
    topo.solute().crossdihedrals().begin(),
    d_to = topo.solute().crossdihedrals().end();

  math::VArray &pos   = conf.current().pos;
  math::VArray &force = conf.current().force;
  math::Vec rcb, rcd, rab, rgf, rgh, ref, rmb, rmf, rng, rnc, ram, rdn, rem, rgn;
  math::Vec rdb, rhf;
  math::Vec fa, fb, fc, fd, fe, ff, fg, fh;
  double dcb2, dgf2, dmb2, dmf2, dnc2, dng2, dam, ddn, ap, dem, dgn, ep, dcb, dgf;
  double energy;
  
  math::Periodicity<B> periodicity(conf.current().box);

  for(int n =0; d_it != d_to; ++d_it, ++n){

    // first calculate the angles: phi
    periodicity.nearest_image(pos(d_it->a), pos(d_it->b), rab);
    periodicity.nearest_image(pos(d_it->c), pos(d_it->b), rcb);
    periodicity.nearest_image(pos(d_it->c), pos(d_it->d), rcd);
    rnc = cross(rcb, rcd);
    dcb2 = abs2(rcb);

    double fram = dot(rab, rcb)/dcb2;
    double frdn = dot(rcd, rcb)/dcb2;

    ram = rab - fram * rcb; 
    rdn = frdn * rcb - rcd;
    dam = sqrt(abs2(ram));
    ddn = sqrt(abs2(rdn));

    ap = dot(ram, rdn); // why not dam and ddn?
    double cosphi = ap / (dam*ddn);
    if ( cosphi > 1) 	cosphi = 1;
    if ( cosphi < -1 )	cosphi = -1;
    double phi = acos(cosphi);
    double sign_phi = dot(rab, rnc);
    if(sign_phi < 0) phi*=-1.0;
    
    // and psi
    periodicity.nearest_image(pos(d_it->e), pos(d_it->f), ref);
    periodicity.nearest_image(pos(d_it->g), pos(d_it->f), rgf);
    periodicity.nearest_image(pos(d_it->g), pos(d_it->h), rgh);
    rng = cross(rgf, rgh);
    dgf2 = abs2(rgf);

    double frem = dot(ref, rgf)/dgf2;
    double frgn = dot(rgh, rgf)/dgf2;

    rem = ref - frem * rgf;
    rgn = frgn * rgf - rgh;
    dem = sqrt(abs2(rem));
    dgn = sqrt(abs2(rgn));

    ep = dot(rem, rgn); // why not dem and dgn?
    double cospsi = ep / (dem*dgn);
    if ( cospsi > 1) 	cospsi = 1;
    if ( cospsi < -1 )	cospsi = -1;
    double psi = acos(cospsi);
    double sign_psi = dot(ref, rng);
    if(sign_psi < 0) psi*=-1.0;

    DEBUG(10, "phi= " << phi << " ,psi= " << psi);

    // now calculate the crossdihedral forces
    rmb = cross(rab, rcb);
    rmf = cross(ref, rgf);
    dmb2 = abs2(rmb);
    dmf2 = abs2(rmf);
    dnc2 = abs2(rnc);
    dng2 = abs2(rng);
   
    assert(unsigned(d_it->type) < param.size());

    double K = param[d_it->type].K;
    double delta = param[d_it->type].pd;
    double m = param[d_it->type].m;
    //double cosdelta = param[d_it->type].cospd;

    DEBUG(10, "crossdihedral K=" << K << " m=" << m << " delta=" << delta);

    double k = K * m * sin(m * (phi + psi) - delta);
    dcb = abs(rcb);
    dgf = abs(rgf);
    double kb1 = dot(rab, rcb)/dcb2 - 1;
    double kb2 = dot(rcd, rcb)/dcb2;
    double kf1 = dot(ref, rgf)/dgf2 - 1;
    double kf2 = dot(rgh, rgf)/dgf2;
    
    fa = k * dcb/dmb2 * rmb;
    fd = -k * dcb/dnc2 * rnc;
    fb = kb1 * fa - kb2 * fd;
    fc = -1.0 * (fa + fb + fd);
    fe = k * dgf/dmf2 * rmf;
    fh = -k * dgf/dng2 * rng;
    ff = kf1 * fe - kf2 * fh;
    fg = -1.0 * (fe + ff + fh);
    
    force(d_it->a) += fa;
    force(d_it->b) += fb;
    force(d_it->c) += fc;
    force(d_it->d) += fd;
    force(d_it->e) += fe;
    force(d_it->f) += ff;
    force(d_it->g) += fg;
    force(d_it->h) += fh;

    // if (V == math::atomic_virial){
    periodicity.nearest_image(pos(d_it->d), pos(d_it->b), rdb);
    periodicity.nearest_image(pos(d_it->h), pos(d_it->f), rhf);

    for(unsigned int i = 0; i < 3; ++i) {
	  for(unsigned int j = 0; j < 3; ++j) {
	    conf.current().virial_tensor(i, j) += rab(i) * fa(j) + rcb(i) * fc(j)
	                                        + rdb(i) * fd(j);
        conf.current().virial_tensor(i, j) += ref(i) * fe(j) + rgf(i) * fg(j)
	                                        + rhf(i) * fh(j);
      }
    }
      DEBUG(11, "\tatomic virial done");
    // }

    // and the energies
    energy = K * (1 + cos(m * (phi + psi) - delta));
    DEBUG(10, "crossdihedral energy= " << energy);
    conf.current().energies.crossdihedral_energy[topo.atom_energy_group()[d_it->a]]
                                                                      += energy;

    // dihedral angle monitoring.
    /*if(sim.param().print.monitor_dihedrals){
      DEBUG(8, "monitoring dihedrals");
      
      DEBUG(11, "dihedral angles: " << phi
	    << " previous minimum: " <<conf.special().dihedral_angle_minimum[n]);
      
      if(fabs(conf.special().dihedral_angle_minimum[n] - phi) > 
	 2*math::Pi / param[d_it->type].m){
	double old_min=conf.special().dihedral_angle_minimum[n];
	conf.special().dihedral_angle_minimum[n] = 
	  _calculate_nearest_minimum(phi, param[d_it->type].m, cosdelta);
	// ugly check to see that it is not the first...
	if(old_min != 4*math::Pi){
	  // could be written to a separate file or by a separate function
	  // should at least be more descriptive.
	  std::cout << "D-A-T: " 
		    << std::setw(4) << topo.solute().atom(d_it->i).residue_nr + 1
		    << std::setw(4) << std::left 
		    << topo.residue_names()[topo.solute().atom(d_it->i).residue_nr]
		    << std::setw(4)  << std::right<< topo.solute().atom(d_it->i).name 
		    << " -"
		    << std::setw(4)  << std::right<< topo.solute().atom(d_it->j).name 
		    << " -"
		    << std::setw(4)  << std::right<< topo.solute().atom(d_it->k).name 
		    << " -"
		    << std::setw(4)  << std::right<< topo.solute().atom(d_it->l).name 
		    << std::setw(6) << d_it->i + 1 << " -"
		    << std::setw(4) << d_it->j + 1 << " -" 
		    << std::setw(4) << d_it->k + 1 << " -"
		    << std::setw(4) << d_it->l + 1
		    << std::setw(10) << std::setprecision(1) 
		    << std::fixed << 180.0*old_min/math::Pi << " -> " 
		    << std::setw(8) << std::setprecision(1) 
		    << std::fixed
		    << 180.0 * conf.special().dihedral_angle_minimum[n]/math::Pi << "\n";
	}
      }
    }*/
  }
  
  return 0;
  
}

int interaction::Crossdihedral_Interaction
::calculate_interactions(topology::Topology &topo,
			 configuration::Configuration &conf,
			 simulation::Simulation &sim)
{
  m_timer.start();

  SPLIT_VIRIAL_BOUNDARY(_calculate_crossdihedral_interactions,
			topo, conf, sim, m_parameter);

  m_timer.stop();

  return 0;
}

/**
 * calculate nearest minimum
 */
static inline double _calculate_nearest_minimum(double phi, int m, double cospd)
{
  // copy from gromos++ nearest_minimum function
  double a_minimum = 0.5*math::Pi*(3.0 - cospd)/ m;
  double delta_phi = 2*math::Pi / m;
  double nearest_min = a_minimum - int(rint((a_minimum - phi)/delta_phi))*delta_phi;
  if(nearest_min >= 2*math::Pi - math::epsilon) nearest_min -= 2*math::Pi;
  
  return nearest_min;
}


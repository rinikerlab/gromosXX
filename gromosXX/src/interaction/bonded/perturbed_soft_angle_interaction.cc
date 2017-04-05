/**
 * @file perturbed_soft_angle_interaction.cc
 * template methods of Perturbed_Soft_Angle_Interaction
 */

#include <climits>
#include "../../stdheader.h"

#include "../../algorithm/algorithm.h"
#include "../../topology/topology.h"
#include "../../simulation/simulation.h"
#include "../../configuration/configuration.h"
#include "../../interaction/interaction.h"

#include "../../math/periodicity.h"

// interactions
#include "../../interaction/interaction_types.h"
#include "perturbed_soft_angle_interaction.h"
#include "../../io/ifp.h"

#include "../../util/template_split.h"
#include "../../util/debug.h"

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE bonded


interaction::Perturbed_Soft_Angle_Interaction::Perturbed_Soft_Angle_Interaction(io::IFP &it)
      : Interaction("PerturbedSoftAngle")
{
      it.read_angles(m_parameter);
}

/**
 * calculate angle forces and energies and lambda derivatives.
 */
template<math::boundary_enum B, math::virial_enum V>
static int _calculate_perturbed_soft_angle_interactions
( topology::Topology & topo,
  configuration::Configuration & conf,
  simulation::Simulation & sim,
  std::vector<interaction::angle_type_struct> &angletypes)
{
  // this is repeated code from Angle_Interaction !!!

  DEBUG(5, "perturbed soft angle interaction");
  DEBUG(7, std::setprecision(5));
  
  // loop over the angles
  std::vector<topology::perturbed_three_body_term_struct>::const_iterator a_it =
    topo.perturbed_solute().softangles().begin(),
    a_to = topo.perturbed_solute().softangles().end();
    
  // and the softness parameters
  std::vector<double>::const_iterator alpha_it =
    topo.perturbed_solute().alpha_angle().begin();


  math::VArray &pos   = conf.current().pos;
  math::VArray &force = conf.current().force;
  math::Vec rij, rkj, fi, fj, fk;

  double energy, e_lambda;

  math::Periodicity<B> periodicity(conf.current().box);

  for( ; a_it != a_to; ++a_it,++alpha_it){

    // atom i determines the energy group for the output. 
    // we use the same definition for the individual lambdas
    const double lambda = topo.individual_lambda(simulation::angle_lambda)
      [topo.atom_energy_group()[a_it->i]][topo.atom_energy_group()[a_it->i]];
    const double lambda_derivative = topo.individual_lambda_derivative
      (simulation::angle_lambda)
      [topo.atom_energy_group()[a_it->i]][topo.atom_energy_group()[a_it->i]];
    DEBUG(7, "soft angle " << a_it->i << "-" << a_it->j << "-" << a_it->k
	  << " A-type " << a_it->A_type
	  << " B-type " << a_it->B_type
	  << " alpha " << *alpha_it
	  << " lambda " << lambda);
	  
    assert(pos.size() > (a_it->i) && pos.size() > (a_it->j) && 
	   pos.size() > (a_it->k));

    periodicity.nearest_image(pos(a_it->i), pos(a_it->j), rij);
    periodicity.nearest_image(pos(a_it->k), pos(a_it->j), rkj);

    double dij = sqrt(abs2(rij));
    double dkj = sqrt(abs2(rkj));
    
    assert(dij != 0.0);
    assert(dkj != 0.0);

    double ip = dot(rij, rkj);
    double cost = ip / (dij * dkj);

    assert(unsigned(a_it->A_type) < angletypes.size());
    assert(unsigned(a_it->B_type) < angletypes.size());
    
    const double K_A = angletypes[a_it->A_type].K;
    const double K_B = angletypes[a_it->B_type].K;
    
    double cos0 =  (1 - lambda) *
      angletypes[a_it->A_type].cos0 +
      lambda *
      angletypes[a_it->B_type].cos0;
    double diff = cost - cos0;
    double diff2 = diff * diff;

    const double K_diff = K_B-K_A;
    const double cos_diff=angletypes[a_it->B_type].cos0- 
      angletypes[a_it->A_type].cos0;
    
    DEBUG(10, "K_A=" << K_A << " K_B=" << K_B << " cos0=" << cos0 << " dij=" << dij << " dkj=" << dkj);
    
    const double soft_A = 1 + *alpha_it * lambda * diff2;
    const double soft_B = 1 + *alpha_it * (1-lambda) * diff2;
    const double soft_A2 = soft_A*soft_A;
    const double soft_B2 = soft_B*soft_B;
    
    
    DEBUG(10, "cost=" << cost << " soft_A=" << soft_A << " soft_B=" << soft_B);

    double fac = ((1-lambda)*K_A / soft_A2 + lambda*K_B / soft_B2) * diff;
    
    fi = -fac * (rkj/dkj - rij/dij * cost) / dij;
    fk = -fac * (rij/dij - rkj/dkj * cost) / dkj;
    fj = -1.0 * fi - fk;
    
    force(a_it->i) += fi;
    force(a_it->j) += fj;
    force(a_it->k) += fk;

    // if (V == math::atomic_virial){
      for(int a=0; a<3; ++a)
	for(int bb=0; bb<3; ++bb)
	  conf.current().virial_tensor(a, bb) += 
	    rij(a) * fi(bb) +
	    rkj(a) * fk(bb);

      DEBUG(11, "\tatomic virial done");
      // }

    energy = 0.5 * ((1-lambda)*K_A / soft_A + lambda*K_B / soft_B) * diff2;

    const double softterm1 = 1 + *alpha_it * diff2;
    const double softterm2 = -2 * *alpha_it * lambda * (1-lambda) * diff * cos_diff;
    
    e_lambda = lambda_derivative 
       * ( 0.5 * diff2 * ( K_A /  soft_A2  * ((-1) * softterm1 - softterm2) 
                        +  K_B /  soft_B2 * (softterm1 - softterm2))
       - diff * cos_diff * ( K_A / soft_A * (1-lambda) + K_B / soft_B * lambda));

    DEBUG(9, "energy: " << energy);

    DEBUG(9, "K_diff: " << K_diff);

    DEBUG(9, "cos_diff: " << cos_diff);
    
    DEBUG(9, "e_lambda: " << e_lambda);
    
    assert(conf.current().energies.angle_energy.size() >
	   topo.atom_energy_group()[a_it->i]);
    
    conf.current().energies.
      angle_energy[topo.atom_energy_group()
		  [a_it->i]] += energy;
    
    assert(conf.current().perturbed_energy_derivatives.angle_energy.size() >
	   topo.atom_energy_group()[a_it->i]);
    
    conf.current().perturbed_energy_derivatives.
      angle_energy[topo.atom_energy_group()
		  [a_it->i]] += e_lambda;

  }

  return 0;
  
}

int interaction::Perturbed_Soft_Angle_Interaction
::calculate_interactions(topology::Topology &topo,
			 configuration::Configuration &conf,
			 simulation::Simulation &sim)
{
  m_timer.start();

  SPLIT_VIRIAL_BOUNDARY(_calculate_perturbed_soft_angle_interactions,
			topo, conf, sim, m_parameter);

  m_timer.stop();

  return 0;
}

int interaction::Perturbed_Soft_Angle_Interaction
::init(topology::Topology &topo, 
		     configuration::Configuration &conf,
		     simulation::Simulation &sim,
		     std::ostream &os,
             bool quiet) 
    {
      if (!quiet)
           os << "Perturbed harmonic soft angle interaction\n";
           
      // add additional angle types with K=0 and the target angle of 
      // the type we are perturbing to or from
      std::vector<topology::perturbed_three_body_term_struct>::iterator bt_it = topo.perturbed_solute().softangles().begin(),
                                 bt_to = topo.perturbed_solute().softangles().end();
      for( ; bt_it != bt_to; ++bt_it){
          if (bt_it->A_type==INT_MAX-1) {
            bt_it->A_type=m_parameter.size();
            double cost = m_parameter[bt_it->B_type].cos0;
            m_parameter.push_back(interaction::angle_type_struct(0, cost));   
            DEBUG(10, "adding new angle type for soft angle perturbation: " 
                       << bt_it->A_type << " K=" << 0 << " cost=" << cost);         
          } else if (bt_it->B_type==INT_MAX-1) {
            bt_it->B_type=m_parameter.size();
            double cost = m_parameter[bt_it->A_type].cos0;
            m_parameter.push_back(interaction::angle_type_struct(0, cost)); 
            DEBUG(10, "adding new angle type for soft angle perturbation: " 
                       << bt_it->B_type << " K=" << 0 << " cost=" << cost);      
          }
      }
      return 0;
    }

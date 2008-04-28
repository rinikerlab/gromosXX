/**
 * @file create_special.cc
 * create the special terms.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

// special interactions
#include <interaction/interaction_types.h>

#include <interaction/special/position_restraint_interaction.h>
#include <interaction/special/distance_restraint_interaction.h>
#include <interaction/special/dihedral_restraint_interaction.h>
#include <interaction/special/perturbed_distance_restraint_interaction.h>
#include <interaction/special/perturbed_dihedral_restraint_interaction.h>
#include <interaction/special/jvalue_restraint_interaction.h>
#include <interaction/special/external_interaction.h>
#include <interaction/special/ramd_interaction.h>

#include <interaction/bonded/dihedral_interaction.h>
#include <interaction/special/pscale.h>

#include <io/instream.h>
#include <io/topology/in_topology.h>

#include "create_special.h"

int interaction::create_special(interaction::Forcefield & ff,
				topology::Topology const & topo,
				simulation::Parameter const & param,
				std::ostream & os,
				bool quiet)
{
  // if (!quiet)
  // os << "SPECIAL\n";
  
  // Position restraints / constraints
  if (param.posrest.posrest == 1 || 
      param.posrest.posrest == 2){

    if(!quiet)
      os <<"\tPosition restraints\n";

    interaction::Position_Restraint_Interaction *pr =
      new interaction::Position_Restraint_Interaction;

    ff.push_back(pr);
    
    if (param.pcouple.virial == math::atomic_virial)
      io::messages.add("Position restraints with atomic virial ill defined",
		       "create_special", io::message::warning);
    else if (param.pcouple.virial == math::molecular_virial)
      io::messages.add("Position restraint forces not added to molecular virial",
		       "create_special", io::message::warning);
  }
  else if (param.posrest.posrest != 0 && param.posrest.posrest != 3){
    io::messages.add("Wrong value for position restraints",
		     "create_special", io::message::error);
  }
  // Distance restraints 
  if (abs(param.distanceres.distanceres) == 1 || 
      abs(param.distanceres.distanceres) == 2){

    if(!quiet)
      os <<"\tDistance restraints\n";

    interaction::Distance_Restraint_Interaction *dr =
      new interaction::Distance_Restraint_Interaction();

    ff.push_back(dr);
    
    if(param.perturbation.perturbation){
      if(!quiet)
	os <<"\tPerturbed distance restraints\n";
      
      interaction::Perturbed_Distance_Restraint_Interaction *pdr =
	new interaction::Perturbed_Distance_Restraint_Interaction;

      ff.push_back(pdr); 
    }
  }
  
  // Dihedral restraints 
  if (param.dihrest.dihrest == 1 || 
      param.dihrest.dihrest == 2){

    if(!quiet)
      os <<"\tDihedral restraints\n";

    interaction::Dihedral_Restraint_Interaction *dr =
      new interaction::Dihedral_Restraint_Interaction();

    ff.push_back(dr);
    
    if(param.perturbation.perturbation){
      if(!quiet)
	os <<"\tPerturbed dihedral restraints\n";
      
      interaction::Perturbed_Dihedral_Restraint_Interaction *pdr =
	new interaction::Perturbed_Dihedral_Restraint_Interaction;
      
      ff.push_back(pdr); 
    }
  }

  // J-Value restraints
  if (param.jvalue.mode != simulation::restr_off){
    if(!quiet){
      os << "\tJ-Value restraints (";
      switch(param.jvalue.mode){
	case simulation::restr_inst :
	  os << "instantaneous";
	  break;
	case simulation::restr_av :
	  os << "time averaged";
	  break;
	case simulation::restr_biq :
	  os << "biquadratic";
	  break;
	default:
	  os << "unknown mode!";
	  break;
      }
      os << ")\n";
    }

    interaction::Jvalue_Restraint_Interaction *jr =
      new interaction::Jvalue_Restraint_Interaction;
    
    ff.push_back(jr);
  }

  // Periodic Scaling
  // right now this only works if no dihedral angles with j-value restraints on are perturbed...
  if (param.pscale.jrest){

    if(!quiet){
      os << "\tscaling based on J-Value restraints\n";
    }
    
    interaction::Periodic_Scaling * ps = 
      new interaction::Periodic_Scaling(ff, param);

    ff.push_back(ps);
  }

  if (param.force.external_interaction){
    if (!quiet){
      os << "\tadding external interaction\n";
    }
    interaction::External_Interaction * ei =
      new interaction::External_Interaction;
    ff.push_back(ei);
  }

  if (param.ramd.fc!=0.0){
    if(!quiet){
      os << "\tadding ramd forces\n";
    }
    interaction::RAMD_Interaction * ri = 
      new interaction::RAMD_Interaction;
    ff.push_back(ri);
  }
  
  return 0;
}

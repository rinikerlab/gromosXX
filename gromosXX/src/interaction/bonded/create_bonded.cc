/**
 * @file create_bonded.cc
 * create the bonded terms.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <math/periodicity.h>

// interactions
#include <interaction/interaction_types.h>
#include <interaction/bonded/quartic_bond_interaction.h>
#include <interaction/bonded/harmonic_bond_interaction.h>
#include <interaction/bonded/angle_interaction.h>
#include <interaction/bonded/dihedral_interaction.h>
#include <interaction/bonded/improper_dihedral_interaction.h>
// perturbed interactions
#include <interaction/bonded/perturbed_quartic_bond_interaction.h>
#include <interaction/bonded/perturbed_harmonic_bond_interaction.h>
#include <interaction/bonded/perturbed_angle_interaction.h>
#include <interaction/bonded/perturbed_improper_dihedral_interaction.h>
#include <interaction/bonded/perturbed_dihedral_interaction.h>

// #include <io/instream.h>
#include <io/ifp.h>

#include "create_bonded.h"

template<math::virial_enum v>
struct bonded_interaction_spec
{
  static const math::virial_enum do_virial = v;
};

template<typename t_interaction_spec>
static void _create_g96_bonded(interaction::Forcefield & ff,
			       topology::Topology const &topo,
			       simulation::Parameter const &param,
			       io::IFP & it,
			       bool quiet = false)
{
  
  if (param.force.bond == 1){
    if (!quiet)
      std::cout <<"\tquartic bond interaction\n";

    interaction::Quartic_Bond_Interaction<t_interaction_spec> *b =
      new interaction::Quartic_Bond_Interaction<t_interaction_spec>();

    it.read_g96_bonds(b->parameter());
    ff.push_back(b);

    if (param.perturbation.perturbation){
      if (!quiet)
	std::cout <<"\tperturbed quartic bond interaction\n";
      
      interaction::Perturbed_Quartic_Bond_Interaction<t_interaction_spec> * pb =
	new interaction::Perturbed_Quartic_Bond_Interaction<t_interaction_spec>(*b);
      ff.push_back(pb);
    }
  }
  else if (param.force.bond == 2){
    if (!quiet)
      std::cout <<"\tharmonic bond interaction\n";

    interaction::Harmonic_Bond_Interaction<t_interaction_spec> *b =
      new interaction::Harmonic_Bond_Interaction<t_interaction_spec>();
    
    it.read_harmonic_bonds(b->parameter());
    ff.push_back(b);

    io::messages.add("using harmonic bond potential", 
		     "create bonded", io::message::notice);

    if (param.perturbation.perturbation){
      if(!quiet)
	std::cout <<"\tperturbed harmonic bond interaction\n";

      interaction::Perturbed_Harmonic_Bond_Interaction<t_interaction_spec> * pb =
	new interaction::Perturbed_Harmonic_Bond_Interaction<t_interaction_spec>(*b);
      ff.push_back(pb);
    }
  }
  
  if (param.force.angle == 1){
    if (!quiet)
      std::cout <<"\tbond angle interaction\n";
    interaction::Angle_Interaction<t_interaction_spec> *a =
      new interaction::Angle_Interaction<t_interaction_spec>();
    
    it.read_angles(a->parameter());
    ff.push_back(a);

    if (param.perturbation.perturbation){
      if (!quiet)
	std::cout <<"\tperturbed bond angle interaction\n";
      interaction::Perturbed_Angle_Interaction<t_interaction_spec> * pa =
	new interaction::Perturbed_Angle_Interaction<t_interaction_spec>(*a);
      ff.push_back(pa);
    }
  }

  if (param.force.angle == 2){
    io::messages.add("harmonic (g87) angle potential not implemented",
		     "create bonded", io::message::error);
  }
  
  if (param.force.improper == 1){
    if (!quiet)
      std::cout << "\timproper dihedral interaction\n";
    
    interaction::Improper_Dihedral_Interaction<t_interaction_spec> * i =
      new interaction::Improper_Dihedral_Interaction<t_interaction_spec>();
    it.read_improper_dihedrals(i->parameter());
    ff.push_back(i);

    if (param.perturbation.perturbation){
      if(!quiet)
	std::cout << "\tperturbed improper dihedral interaction\n";
      interaction::Perturbed_Improper_Dihedral_Interaction<t_interaction_spec> * pi =
	new interaction::Perturbed_Improper_Dihedral_Interaction<t_interaction_spec>(*i);
      ff.push_back(pi);
    }

  }

  if (param.force.dihedral == 1){
    if (!quiet)
      std::cout <<"\tdihedral interaction\n";
    
    interaction::Dihedral_Interaction<t_interaction_spec> * d =
      new interaction::Dihedral_Interaction<t_interaction_spec>();
    it.read_dihedrals(d->parameter());
    ff.push_back(d);

    if (param.perturbation.perturbation){
      if(!quiet)
	std::cout <<"\tperurbed dihedral interaction\n";
      interaction::Perturbed_Dihedral_Interaction<t_interaction_spec> * pd =
	new interaction::Perturbed_Dihedral_Interaction<t_interaction_spec>(*d);
      ff.push_back(pd);
    }

  }
  
}

int interaction::create_g96_bonded(interaction::Forcefield & ff,
				   topology::Topology const & topo,
				   simulation::Parameter const & param,
				   io::IFP & it, bool quiet)
{
  switch(param.pcouple.virial){
    case math::no_virial:
      {
	// create an interaction spec suitable for the bonded terms
	_create_g96_bonded<bonded_interaction_spec<math::no_virial> >
	  (ff, topo, param, it, quiet);
	break;
      }
    case math::atomic_virial:
      {
	// create an interaction spec suitable for the bonded terms
	_create_g96_bonded<bonded_interaction_spec<math::atomic_virial> >
	  (ff, topo, param, it, quiet);
	break;
      }
    case math::molecular_virial:
      {
	// create an interaction spec suitable for the bonded terms
	_create_g96_bonded<bonded_interaction_spec<math::molecular_virial> >
	  (ff, topo, param, it, quiet);
	break;
      }
    default:
      {
	throw std::string("Wrong virial type requested");
      }
  }
  return 0;
  
}


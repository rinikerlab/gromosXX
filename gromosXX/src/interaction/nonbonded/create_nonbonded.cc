/**
 * @file create_nonbonded.cc
 * create the nonbonded interaction.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <interaction/interaction.h>
#include <interaction/interaction_types.h>
#include <interaction/nonbonded/interaction/nonbonded_parameter.h>

#include <interaction/forcefield/forcefield.h>

#include <interaction/nonbonded/interaction/storage.h>

#include <interaction/nonbonded/pairlist/pairlist.h>
#include <interaction/nonbonded/pairlist/pairlist_algorithm.h>
#include <interaction/nonbonded/pairlist/standard_pairlist_algorithm.h>
#include <interaction/nonbonded/pairlist/grid_pairlist_algorithm.h>

#include <interaction/nonbonded/interaction/nonbonded_outerloop.h>
#include <interaction/nonbonded/interaction/nonbonded_set.h>

#include <interaction/nonbonded/interaction/nonbonded_interaction.h>

#include <io/ifp.h>

#include <interaction/nonbonded/create_nonbonded.h>

#include <util/debug.h>


#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE nonbonded

using namespace std;

int interaction::create_g96_nonbonded
(
 interaction::Forcefield & ff,
 topology::Topology const & topo,
 simulation::Simulation const & sim,
 configuration::Configuration const & conf,
 io::IFP & it,
 std::ostream & os,
 bool quiet
 )
{
  if (sim.param().force.nonbonded == 0) return 0;
  
  if(!quiet){
    if (!sim.param().pairlist.grid)
      os << "\t" << setw(20) << left << "PairlistAlgorithm" << setw(30) 
	   << left << "Standard_Pairlist_Algorithm" << right << "\n";
    else{
      os << "\t" << setw(20) << left << "PairlistAlgorithm" << setw(30) 
	   << left << "Grid_Pairlist_Algorithm" << right << "\n";
    }
    
    if (sim.param().pairlist.atomic_cutoff)
      os << "\t" << setw(20) << left << "atomic-cutoff" << setw(30) << left << "on" << right << "\n";
    else
      os << "\t" << setw(20) << left << "atomic-cutoff" << setw(30) << left << "off" << right << "\n";

    switch(sim.param().boundary.boundary){
      case math::vacuum:
	os << "\t" << setw(20) << left << "boundary" << setw(30) 
	     << left << "vacuum" << right << "\n";
	break;
      case math::rectangular:
	os << "\t" << setw(20) << left << "boundary" << setw(30) 
	     << left << "rectangular" << right << "\n";
	break;
      case math::truncoct:
	os << "\t" << setw(20) << left << "boundary" << setw(30) 
	     << left << "truncoct" << right << "\n";
	break;
      case math::triclinic:
	os << "\t" << setw(20) << left << "boundary" << setw(30) 
	     << left << "triclinic" << right << "\n";
	break;
      default:
	os << "\t" << setw(20) << left << "boundary" << setw(30) 
	     << left << "unknown" << right << "\n";
    }

    switch(sim.param().pcouple.virial){
      case math::no_virial:
	os << "\t" << setw(20) << left << "virial" << setw(30) << left << "none" << right << "\n";
	break;
      case math::molecular_virial:
	os << "\t" << setw(20) << left << "virial" << setw(30) << left << "molecular" << right << "\n";
	break;
      case math::atomic_virial:
	os << "\t" << setw(20) << left << "virial" << setw(30) << left << "atomic" << right << "\n";
	break;
      default:
	os << "\t" << setw(20) << left << "virial" << setw(30) << left << "unknown" << right << "\n";
    }
    
    if (sim.param().perturbation.perturbation){
      os << "\t" << setw(20) << left << "perturbation" << setw(30) << left << "on" << right << "\n";
      if (sim.param().perturbation.scaling)
	os << "\t" << setw(20) << left << "scaling" << setw(30) << left << "on" << right << "\n";
      else
	os << "\t" << setw(20) << left << "scaling" << setw(30) << left << "off" << right << "\n";
    }
    else{
      os << "\t" << setw(20) << left << "perturbation" << setw(30) << left << "off" << right << "\n";
    }

  }
  
  Pairlist_Algorithm * pa;
  if (!sim.param().pairlist.grid){
    pa = new Standard_Pairlist_Algorithm();
  }
  else{
    pa = new Grid_Pairlist_Algorithm();
  }
  
  Nonbonded_Interaction * ni = new Nonbonded_Interaction(pa);
  it.read_lj_parameter(ni->parameter().lj_parameter());

  pa->set_parameter(&ni->parameter());
  
  ff.push_back(ni);

  if (!quiet){
    os
      << "\t\t\tshortrange cutoff      : "
      << sim.param().pairlist.cutoff_short << "\n"
      << "\t\t\tlongrange cutoff       : "
      << sim.param().pairlist.cutoff_long << "\n"
      << "\t\t\tepsilon                : "
      << sim.param().longrange.epsilon << "\n"
      << "\t\t\treactionfield epsilon  : "
      << sim.param().longrange.rf_epsilon << "\n"
      << "\t\t\tkappa                  : "
      << sim.param().longrange.rf_kappa << "\n"
      << "\t\t\treactionfield cutoff   : "
      << sim.param().longrange.rf_cutoff << "\n"
      << "\t\t\tpairlist creation every "
      << sim.param().pairlist.skip_step
      << " steps\n\n";
  }
  
  return 0;
}

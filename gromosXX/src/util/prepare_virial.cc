/**
 * prepare for virial calculation.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <math/periodicity.h>

#include "prepare_virial.h"

#undef MODULE
#undef SUBMODULE

#define MODULE util
#define SUBMODULE util

#include <util/debug.h>

template<math::boundary_enum b>
static void _center_of_mass(topology::Atom_Iterator start, 
			    topology::Atom_Iterator end,
			    topology::Topology const & topo,
			    configuration::Configuration const & conf,
			    math::Vec &com_pos, 
			    math::Matrix &com_e_kin,
			    math::Periodicity<b> const & periodicity)
{

  com_pos = 0.0;
  double m;
  double tot_mass = 0.0;

  math::Vec p;
  math::Vec prev;
  math::Vec v = 0.0;

  prev = conf.current().pos(*start);

  for( ; start != end; ++start){

    assert(unsigned(topo.mass().size()) > *start &&
           unsigned(conf.current().pos.size()) > *start);

    m = topo.mass()(*start);
    tot_mass += m;
    periodicity.nearest_image(conf.current().pos(*start), prev, p);
    com_pos += m * (p + prev);
    v += m * conf.current().vel(*start);
    prev += p;
  }

  com_pos /= tot_mass;

  for(int i=0; i<3; ++i)
    for(int j=0; j<3; ++j)
      com_e_kin(i,j) = 0.5 * v(i) * v(j) / tot_mass;

}


template<math::boundary_enum b>
static void _prepare_virial(topology::Topology const & topo,
			    configuration::Configuration & conf,
			    simulation::Simulation const & sim)
{
  if (sim.param().pcouple.virial == math::molecular_virial){

    DEBUG(10, "lambda = " << topo.lambda());
    
    math::Periodicity<b> periodicity(conf.current().box);

    topology::Molecule_Iterator
      m_it = topo.molecule_begin(),
      m_to = topo.molecule_end();

    math::Vec com_pos;
    math::Matrix com_ekin;

    conf.current().kinetic_energy_tensor = 0.0;

    for( ; m_it != m_to; ++m_it){
      _center_of_mass(m_it.begin(),
		      m_it.end(),
		      topo, conf,
		      com_pos, com_ekin,
		      periodicity);

      for(int i=0; i<3; ++i)
	for(int j=0; j<3; ++j)
	  conf.current().kinetic_energy_tensor(i,j) += com_ekin(i,j);

      topology::Atom_Iterator a_it = m_it.begin(),
	a_to = m_it.end();

      math::VArray const &pos = conf.current().pos;

      for( ; a_it != a_to; ++a_it){
	assert(unsigned(conf.special().rel_mol_com_pos.size()) > *a_it);
	periodicity.nearest_image(pos(*a_it), com_pos,
				  conf.special().rel_mol_com_pos(*a_it));
      }
      
    }
    
  }
  
  else if (sim.param().pcouple.virial == math::atomic_virial){

    conf.current().kinetic_energy_tensor = 0.0;

    for(unsigned int i=0; i < topo.num_atoms(); ++i){
      for(int a=0; a<3; ++a){
	for(int bb=0; bb<3; ++bb){
	  conf.current().kinetic_energy_tensor(a, bb) +=
	    0.5 * topo.mass()(i) *
	    conf.current().vel(i)(a) * 
	    conf.current().vel(i)(bb);
	}
      }
    }

    // system().molecular_kinetic_energy() *= 0.5;
  }

}


void util::prepare_virial(topology::Topology const & topo,
			  configuration::Configuration & conf,
			  simulation::Simulation const & sim)
{
  switch(conf.boundary_type){
    case math::vacuum :
      // no virial necessary!!!
      break;
    case math::triclinic :
      _prepare_virial<math::triclinic> (topo, conf, sim);
      break;
    case math::rectangular :
      _prepare_virial<math::rectangular> (topo, conf, sim);
      break;
    default:
      throw std::string("Wrong boundary type");
  }
  
}

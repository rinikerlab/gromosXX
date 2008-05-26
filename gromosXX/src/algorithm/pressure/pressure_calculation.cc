/**
 * @file pressure_calculation.cc
 * calculates the pressure.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <configuration/state_properties.h>

#include <math/volume.h>

#include "pressure_calculation.h"

#undef MODULE
#undef SUBMODULE

#define MODULE algorithm
#define SUBMODULE temperature

#include <util/debug.h>

int algorithm::Pressure_Calculation
::apply(topology::Topology & topo,
	configuration::Configuration & conf,
	simulation::Simulation & sim)
{
  DEBUG(7, "Pressure calculation");

  m_timer.start();

  // calculate the pressure tensor
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j){
      // virial -0.5 factor applied here!
      conf.old().virial_tensor(i,j) *= -0.5;
      conf.old().pressure_tensor(i,j) = 2 * 
	(conf.old().kinetic_energy_tensor(i,j)
	 - conf.old().virial_tensor(i,j)) /
	math::volume(conf.old().box, conf.boundary_type);
    }
  }
  
  m_timer.stop();
  
  return 0;
  
}

/**
 * @file leap_frog.cc
 * contains the implementation
 * for the classes Leap_Frog_Position and Leap_Frog_Velocity.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include "leap_frog.h"

#undef MODULE
#undef SUBMODULE
#define MODULE algorithm
#define SUBMODULE integration

/**
 * Leap frog step.
 */
int algorithm::Leap_Frog_Position
::apply(topology::Topology & topo,
	configuration::Configuration & conf,
	simulation::Simulation &sim)
{
  const double start = util::now();
  
  const int num_atoms = topo.num_atoms();
  
  // r = r + v*dt
  for(int i=0; i < num_atoms; ++i)
    conf.current().pos(i) =
      conf.old().pos(i) + conf.current().vel(i) * sim.time_step_size();
  
  m_timing += util::now() - start;
  
  return 0;
}

/**
 * Leap frog step.
 */
int algorithm::Leap_Frog_Velocity
::apply(topology::Topology & topo,
	configuration::Configuration & conf,
	simulation::Simulation &sim)
{
  const double start = util::now();

  conf.exchange_state();
  // copy the box
  conf.current().box = conf.old().box;

  const int num_atoms = topo.num_atoms();

  // v = v + f * dt / m
  for(int i=0; i < num_atoms; ++i){
    conf.current().vel(i) =
      conf.old().vel(i) + conf.old().force(i) * sim.time_step_size() / topo.mass()(i);

    DEBUG(10, "atom " << i
	  << "\n\tf=" << math::v2s(conf.old().force(i))
	  << "\n\tmass=" << topo.mass()(i)
	  << "\n\tvel=" << math::v2s(conf.old().vel(i)));
  }
  
  m_timing += util::now() - start;
  
  return 0;
  
}

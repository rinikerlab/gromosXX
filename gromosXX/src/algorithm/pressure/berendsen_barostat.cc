/**
 * @file berendsen_barostat.cc
 * methods of the berendsen barostat.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <configuration/state_properties.h>

#include "berendsen_barostat.h"

#undef MODULE
#undef SUBMODULE

#define MODULE algorithm
#define SUBMODULE pressure

#include <util/debug.h>

int algorithm::Berendsen_Barostat
::apply(topology::Topology & topo,
	configuration::Configuration & conf,
	simulation::Simulation & sim)
{
  const double start = util::now();
  
  DEBUG(8, "Berendsen Barostat == apply");

  // position are current!
  math::VArray & pos = conf.current().pos;
  math::Matrix & pressure = conf.old().pressure_tensor;
  math::Box & box = conf.current().box;

  DEBUG(9, "scaling = " << sim.param().pcouple.scale);
  
  switch(sim.param().pcouple.scale){
    case math::pcouple_isotropic:
      {
	DEBUG(9, "total pressure (isotropic) ...");
	double total_pressure =  (pressure(0,0)
				  + pressure(1,1)
				  + pressure(2,2)) / 3.0;

	DEBUG(8, "pressure: " << total_pressure);
	
	double mu = pow(1.0 - sim.param().pcouple.compressibility
			* sim.time_step_size() / sim.param().pcouple.tau
			* (sim.param().pcouple.pres0(0,0) - total_pressure),
			1.0/3.0);

	DEBUG(8, "mu: " << mu);

	// scale the box
	box *= mu;

	// scale the positions
	for(unsigned int i=0; i<pos.size(); ++i)
	  pos(i) = mu * pos(i);

	break;
      }
    case math::pcouple_anisotropic:
      {
	math::Vec mu;

	DEBUG(8, "anisotropic pressure scaling");

	for(int i=0; i<3; ++i){
	  mu(i) = pow(1.0 - sim.param().pcouple.compressibility
		      * sim.time_step_size() / sim.param().pcouple.tau
		      * (sim.param().pcouple.pres0(i,i) - 
			 pressure(i,i)),
		      1.0/3.0);
	}

	DEBUG(10, "mu = " << math::v2s(mu));
	
	// scale the box
	for(int i=0; i<3; ++i)
	  for(int j=0; j<3; ++j)
	    box(i)(j) *= mu(j);

	DEBUG(10, "and the positions...");

	// scale the positions
	for(unsigned int i=0; i<pos.size(); ++i)
	  for(int j=0; j<3; ++j)
	    pos(i)(j) *= mu(j);
	
	break;
      }
    case math::pcouple_full_anisotropic:
      {
	
	math::Matrix mu;

	for(int i=0; i<3; ++i){
	  for(int j=0; j<3; ++i){
	  
	    mu(i, j) = pow(1.0 - sim.param().pcouple.compressibility
			   * sim.time_step_size() / sim.param().pcouple.tau
			   * (sim.param().pcouple.pres0(i,j) -
			      pressure(i,j)),
			   1.0/3.0);
	  }
	}

	// scale the box
	box = math::product(mu, box);
	
	// scale the positions
	for(unsigned int i=0; i<pos.size(); ++i)
	  pos(i) = math::product(mu, pos(i));

      }
    default:
      return 0;
  }

  m_timing += util::now() - start;

  return 0;
  
}

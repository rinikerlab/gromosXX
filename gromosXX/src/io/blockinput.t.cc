/**
 * @file blockinput.t.cc
 * test routines for blockinput.
 */

#include "../debug.h"

#include "../math/gmath.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <stdexcept>

#include "../simulation/simulation.h"
#include "../interaction/interaction.h"

#include "io.h"

#include "../debug.cc"

/**
 * test the io.
 */
int test_blockio()
{
  int result = 0;

  simulation::system the_system;
  simulation::topology the_topology;
  
  typedef simulation::simulation<simulation::topology, simulation::system>
    simulation_type;  

  simulation_type the_simulation(the_topology, the_system);

  interaction::forcefield<simulation_type> the_forcefield;

  interaction::harmonic_bond_interaction<simulation_type> *bond_interaction
    = new interaction::harmonic_bond_interaction<simulation_type>;
  
  std::ifstream topo_file("/home/markus/test/hexa10.topo");
  if (!topo_file.good()){
    std::cout << "could not open topology: /home/markus/test/hexa10.topo" << std::endl;
    return 1;
  }
  
  io::InTopology topo(topo_file);
  
  std::ifstream sys_file("/home/markus/test/hexa10.coord");
  if (!sys_file.good()){
    std::cout << "could not open system: /home/markus/test/hexa10.coord" << std::endl;
    return 2;
  }
  io::InTrajectory sys(sys_file);

  // std::cout << "reading bond interaction" << std::endl;
  topo >> *bond_interaction;
  
  // std::cout << "reading topology" << std::endl;
  topo >> the_topology;
  
  // std::cout << "reading system" << std::endl;
  sys >> the_system;

  if (io::message::notice != io::messages.display()) ++result;

  // output
  std::ofstream final("blockinput.t.fin");
  std::ofstream trj("blockinput.t.trj");
  
  io::OutTrajectory<simulation_type> traj(trj, final);
  traj << the_simulation;
  // traj << io::decorated << the_simulation;
  traj << io::final << the_simulation;

  return result;

}

int main()
{
  int r1;
  try{
    if ((r1 = test_blockio()))
      std::cout << "test_blockio failed" << std::endl;
  }
  catch(std::runtime_error e){
    std::cout << "Exception in test_blockio\n" << e.what() << std::endl;
    throw;
  }
  
  return r1;
}


/**
 * @file src/simulation/simulation/simulation.h
 * the simulation class
 */

#ifndef INCLUDED_SIMULATION_H
#define INCLUDED_SIMULATION_H

/**
 * @namespace simulation
 * namespace that contains the simulation class
 * and the associated topology, system and parameter clases
 */
namespace simulation
{
  /**
   * @class simulation
   * holds together the system, the topology and
   * input parameters.
   * provides the simulation properties.
   */
  template<typename t_topo, typename t_system>
  class simulation
  {
  public:
    typedef t_topo topology_type;
    typedef t_system system_type;
    
    /**
     * Constructor
     */
    explicit simulation(t_topo &topo, t_system &sys);
    
    /**
     * topology accessor
     */
    t_topo & topology();
    /**
     * system accessor
     */
    t_system & system();
    /**
     * time accessor
     */
    double time();
    /**
     * steps accessor
     */
    int steps();
    
  private:
    /**
     * the topology.
     */
    topology_type &m_topology;
    /** 
     * the system.
     */
    system_type   &m_system;
    /**
     * the time.
     */
    double m_time;
    /**
     * the number of steps done.
     */
    int m_steps;

  }; // class simulation
  
  
} // namespace simulation

// template definition
#include "simulation.tcc"

#endif

/**
 * @file src/simulation/simulation/simulation.h
 * the Simulation class
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
   * @class Simulation
   * holds together the system, the topology and
   * input parameters.
   * provides the simulation properties.
   */
  template<typename t_topo, typename t_system>
  class Simulation
  {
  public:
    typedef t_topo topology_type;
    typedef t_system system_type;
    
    /**
     * Constructor
     */
    explicit Simulation(t_topo &topo, t_system &sys);
    
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
     * set (initial) time.
     */
    void time(double t);
    /**
     * steps accessor
     */
    int steps();
    /**
     * nonbonded parameters
     */
    simulation::Nonbonded nonbonded();

    /**
     * increase the time by dt.
     */
    void increase_time(double dt);

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

    /**
     * nonbonded parameters.
     */
    simulation::Nonbonded m_nonbonded;
    
  }; // class Simulation
  
  
} // namespace simulation

// template definition
#include "simulation.tcc"

#endif

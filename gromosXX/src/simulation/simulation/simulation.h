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
    explicit Simulation();
    
    /**
     * const topology accessor
     */
    t_topo const & topology()const;

    /**
     * topology accessor
     */
    t_topo & topology();
    /**
     * system accessor
     */
    t_system & system();
    /**
     * const system accessor
     */
    t_system const & system()const;
    
    /**
     * time accessor
     */
    double time();
    /**
     * old time accessor
     */
    double old_time();
    
    /**
     * set (initial) time.
     */
    void time(double t);
    /**
     * steps accessor
     */
    int steps();
    /**
     * const nonbonded parameters
     */
    simulation::Nonbonded const & nonbonded()const;

    /**
     * nonbonded parameters
     */
    simulation::Nonbonded & nonbonded();

    /**
     * the multibath / degree of freedom parameters
     */
    simulation::Multibath const & multibath()const;

    /**
     * the multibath / degree of freedom parameters
     */
    simulation::Multibath & multibath();
    
    /**
     * increase the time by dt.
     */
    void increase_time(double dt);

    /**
     * add solvent molecules to the simulation (system).
     */
    void solvate(size_t solv, size_t num_molecules);

    /**
     * calculate degrees of freedom.
     */
    void calculate_degrees_of_freedom();

    /**
     * put chargegroups into the central (computational) box.
     * update the box indices.
     */
    void put_chargegroups_into_box();
    /**
     * const calculate virial
     */
    const bool pressure_calculation()const;

    /**
     * calculate virial
     */
    bool pressure_calculation();
    
    void pressure_calculation(bool pc);
    
    /**
     * calculate positions relative to molecular com
     */
    void calculate_mol_com();
    
    /**
     * calculate molecular kinetic energies.
     * internal and rotational and translational.
     */
    void calculate_mol_ekin(int mean = 0);

    /**
     * check the state of the class (class invariant)
     * @return 0 if ok.
     */
    int check_state()const;
    
  private:
    /**
     * the topology.
     */
    topology_type m_topology;
    /** 
     * the system.
     */
    system_type   m_system;
    /**
     * the time.
     */
    double m_time;
    /**
     * the time of the previous step
     */
    double m_old_time;
    
    /**
     * the number of steps done.
     */
    int m_steps;
    /**
     * nonbonded parameter.
     */
    simulation::Nonbonded m_nonbonded;
    
    /**
     * multibath parameter.
     */
    simulation::Multibath m_multibath;

    /**
     * do we calculate the pressure?
     */
    bool m_pressure_calculation;
    
  }; // class Simulation
  
  
} // namespace simulation

// template definition
#include "simulation.tcc"

#endif

/**
 * @file system.h
 * the system class
 */

#ifndef INCLUDED_SYSTEM_H
#define INCLUDED_SYSTEM_H

namespace simulation
{
  /**
   * @class system
   * holds the state information of
   * the simulated system.
   * @sa simulation::simulation
   * @sa simulation::topology
   */
  class system
  {
  public:
    /**
     * Constructor
     */
    explicit system();
    /**
     * set the number of atoms in the system.
     */
    void resize(size_t s);
    /**
     * position accessor
     */
    math::VArray &pos();
    /**
     * old position
     */
    math::VArray const &old_pos()const;
    /**
     * velocity accessor
     */
    math::VArray &vel();
    /**
     * old velocity
     */
    math::VArray const &old_vel()const;
    /**
     * force accessor
     */
    math::VArray &force();
    /**
     * old force
     */
    math::VArray const &old_force()const;
    /**
     * exchange positions
     */
    void exchange_pos();
    /**
     * exchange velocities
     */
    void exchange_vel();
    /**
     * exchage forces
     */
    void exchange_force();
  protected:
    /**
     * position 1
     */
    math::VArray m_position1;
    /**
     * position 2
     */
    math::VArray m_position2;
    /**
     * velocity 1
     */
    math::VArray m_velocity1;
    /**
     * velocity 2
     */
    math::VArray m_velocity2;
    /**
     * force 1
     */
    math::VArray m_force1;
    /**
     * force 2
     */
    math::VArray m_force2;
    /**
     * position
     */
    math::VArray *m_pos;
    /**
     * old position
     */
    math::VArray *m_old_pos;
    /**
     * velocity
     */
    math::VArray *m_vel;
    /**
     * old velocity
     */
    math::VArray *m_old_vel;
    /**
     * force
     */
    math::VArray *m_force;
    /**
     * old force
     */
    math::VArray *m_old_force;
    
  }; // system
  
} // simulation

// template and inline definitions
#include "system.tcc"

#endif

  

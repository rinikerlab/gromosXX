/**
 * @file simulation.tcc
 * contains the (template) methods
 * for the class Simulation
 */

/**
 * Constructor
 */
template<typename t_topo, typename t_system>
inline simulation::Simulation<t_topo, t_system>
::Simulation(t_topo &topo, t_system &sys)
  : m_topology(topo),
    m_system(sys),
    m_time(0.0),
    m_old_time(0.0),
    m_steps(0)
{
}

/**
 * const topology accessor
 */
template<typename t_topo, typename t_system>
inline t_topo const & simulation::Simulation<t_topo, t_system>
::topology()const
{
  return m_topology;
}

/**
 * topology accessor
 */
template<typename t_topo, typename t_system>
inline t_topo & simulation::Simulation<t_topo, t_system>
::topology()
{
  return m_topology;
}

/**
 * const system accessor
 */
template<typename t_topo, typename t_system>
inline t_system const & simulation::Simulation<t_topo, t_system>
::system()const
{
  return m_system;
}

/**
 * system accessor
 */
template<typename t_topo, typename t_system>
inline t_system & simulation::Simulation<t_topo, t_system>
::system()
{
  return m_system;
}

/**
 * time accessor
 */
template<typename t_topo, typename t_system>
inline double simulation::Simulation<t_topo, t_system>
::time()
{
  return m_time;
}

/**
 * old time accessor
 */
template<typename t_topo, typename t_system>
inline double simulation::Simulation<t_topo, t_system>
::old_time()
{
  return m_old_time;
}

/**
 * set time
 */
template<typename t_topo, typename t_system>
inline void simulation::Simulation<t_topo, t_system>
::time(double t)
{
  m_time = t;
}

/**
 * increase time by dt
 */
template<typename t_topo, typename t_system>
inline void simulation::Simulation<t_topo, t_system>
::increase_time(double dt)
{
  m_old_time=m_time;
  m_time += dt;
  ++m_steps;
}

/**
 * steps accessor
 */
template<typename t_topo, typename t_system>
inline int simulation::Simulation<t_topo, t_system>
::steps()
{
  return m_steps;
}

/**
 * Nonbonded interaction class
 */
template<typename t_topo, typename t_system>
inline simulation::Nonbonded & simulation::Simulation<t_topo, t_system>
::nonbonded()
{
  return m_nonbonded;
}

/**
 * const nonbonded interaction class
 */
template<typename t_topo, typename t_system>
inline simulation::Nonbonded const & simulation::Simulation<t_topo, t_system>
::nonbonded()const
{
  return m_nonbonded;
}

/**
 * multibath parameter
 */
template<typename t_topo, typename t_system>
inline simulation::Multibath const & simulation::Simulation<t_topo, t_system>
::multibath()const
{
  return m_multibath;
}

/**
 * multibath parameter
 */
template<typename t_topo, typename t_system>
inline simulation::Multibath & simulation::Simulation<t_topo, t_system>
::multibath()
{
  return m_multibath;
}

/**
 * add solvent molecules to the simulation (system).
 */
template<typename t_topo, typename t_system>
inline void simulation::Simulation<t_topo, t_system>::solvate(size_t solv, size_t num_molecules)
{
  topology().solvate(solv, num_molecules);
  system().resize(topology().num_solute_atoms() + 
		  topology().num_solvent_atoms());
}

/**
 * calculate the degrees of freedom.
 */
template<typename t_topo, typename t_system>
inline void simulation::Simulation<t_topo, t_system>::calculate_degrees_of_freedom()
{
  multibath().calculate_degrees_of_freedom(topology());
  system().energies().kinetic_energy.resize(multibath().size());
}

/**
 * put chargegroups into box while updating the box indices for the atoms.
 */
template<typename t_topo, typename t_system>
inline void simulation::Simulation<t_topo, t_system>::put_chargegroups_into_box()
{
  io::messages.add("not implemented","Simulation: put_chargegroups_into_box",
		   io::message::error);
}

template<typename t_topo, typename t_system>
inline int simulation::Simulation<t_topo, t_system>::check_state()const
{
  DEBUG(7, "checking state of Simulation");

  int result = 0;

  // check associated classes
  result += m_topology.check_state();
  result += m_system.check_state();

  result += m_nonbonded.check_state();
  result += m_multibath.check_state(m_topology.num_atoms());

  if (m_time < 0)
    io::messages.add("Simulation time < 0", "Simulation::check_state",
		     io::message::warning);
  if (m_steps < 0){
    io::messages.add("Simulation steps < 0", "Simulation::check_state",
		     io::message::error);
    ++result;
  }

  return result;
  
}

namespace simulation
{
  template<typename t_topo, typename t_system>
  std::ostream &operator<<(std::ostream &os, Simulation<t_topo, t_system> &sys)
  {
    os << "a simulation";
    return os;
  }
}

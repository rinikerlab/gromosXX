/**
 * @file perturbed_shake.h
 * the perturbed shake algorithm.
 */

#ifndef INCLUDED_PERTURBED_SHAKE_H
#define INCLUDED_PERTURBED_SHAKE_H

namespace algorithm
{
  /**
   * @class Perturbed_Shake
   * implements the shake algorithm for perturbed distance constraints.
   */
  template<math::virial_enum do_virial>
  class Perturbed_Shake : public Shake<do_virial>
  {
  public:
    /**
     * Constructor.
     */
    Perturbed_Shake(double const tolerance = 0.000001,
		    int const max_iterations = 1000);
    
    /**
     * Destructor.
     */
    virtual ~Perturbed_Shake();
        
    /**
     * apply perturbed SHAKE
     * (also calls SHAKE for the unperturbed bonds)
     */
    virtual int apply(topology::Topology & topo,
		      configuration::Configuration & conf,
		      simulation::Simulation & sim);
    
    /**
     * initialize startup positions and velocities
     * if required.
     */
    int init(topology::Topology & topo,
	     configuration::Configuration & conf,
	     simulation::Simulation & sim,
	     bool quiet = false);

  protected:

    /**
     * do one iteration
     */
    template<math::boundary_enum b>
    int perturbed_shake_iteration(topology::Topology const &topo,
				  configuration::Configuration & conf,
				  bool & convergence,
				  int const first,
				  std::vector<bool> &skip_now,
				  std::vector<bool> &skip_next,
				  std::vector<topology::perturbed_two_body_term_struct>
				  const & constr,
				  double const dt,
				  math::Periodicity<b> const & periodicity,
				  bool do_constraint_force = false,
				  size_t force_offset = 0);
    

    /**
     * shake (perturbed) solute
     */
    template<math::boundary_enum b>
    int perturbed_solute(topology::Topology const & topo,
			 configuration::Configuration & conf,
			 double dt, int const max_iterations);

    // overwrites the other one, as g++ seems unable to compile this...!!!
    /**
     * shake solvent (not perturbed)
     */
    template<math::boundary_enum b>
    int solvent(topology::Topology const & topo,
		configuration::Configuration & conf,
		double dt, int const max_iterations);

  };
  
} //algorithm

// template methods
#include "perturbed_shake.cc"

#endif

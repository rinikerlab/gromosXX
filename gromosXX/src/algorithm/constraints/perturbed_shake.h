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
  class Perturbed_Shake : public Shake
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
    template<math::boundary_enum B, math::virial_enum V>
    int perturbed_shake_iteration(topology::Topology const &topo,
				  configuration::Configuration & conf,
				  bool & convergence,
				  int first,
				  std::vector<bool> &skip_now,
				  std::vector<bool> &skip_next,
				  std::vector<topology::perturbed_two_body_term_struct>
				  const & constr,
				  double dt,
				  math::Periodicity<B> const & periodicity,
				  bool do_constraint_force = false,
				  unsigned int force_offset = 0);
    

    /**
     * shake (perturbed) solute
     */
    template<math::boundary_enum B, math::virial_enum V>
    void perturbed_solute(topology::Topology const & topo,
			  configuration::Configuration & conf,
			  double dt, int max_iterations,
			  int & error);

    // overwrites the other one, as g++ seems unable to compile this...!!!
    /**
     * shake solvent (not perturbed)
     */
    template<math::boundary_enum B, math::virial_enum V>
    void solvent(topology::Topology const & topo,
		 configuration::Configuration & conf,
		 double dt, int const max_iterations,
		 int & error);

  };
  
} //algorithm

#endif

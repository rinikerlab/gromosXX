/**
 * @file interaction_spec.h
 * specification of how to calculate the interactions
 */

#ifndef INCLUDED_INTERACTION_SPEC_H
#define INCLUDED_INTERACTION_SPEC_H

namespace interaction
{
  const bool atomic_cutoff_on = true;
  const bool atomic_cutoff_off = false;
  const bool perturbation_on = true;
  const bool perturbation_off = false;
  const bool scaling_on = true;
  const bool scaling_off = false;
  
  /**
   * @class Interaction_Spec
   * interaction specifications.
   */
  template<
    math::boundary_enum t_boundary = math::rectangular,
    math::virial_enum t_virial = math::molecular_virial
    >
  class Interaction_Spec
  {
  public:
    typedef Interaction_Spec<t_boundary,
			     t_virial
			     >
    interaction_spec_type;

    static const math::boundary_enum boundary_type = t_boundary;
    static const math::virial_enum do_virial = t_virial;
    
  };

  /**
   * @class Perturbation_Spec
   * specifies if and what kind of perturbation to do (or rather apply)
   */
  template<
    bool t_scaling = scaling_off
  >
  class Perturbation_Spec
  {
  public:
    typedef Perturbation_Spec<t_scaling>
    perturbation_spec_type;
    
    static const bool do_scaling = t_scaling;
  };


} // interaction

#endif

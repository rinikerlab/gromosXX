/**
 * @file innerloop_template.h
 * call a (member) function with the correct template parameters to
 * define an Nonbonded_Innerloop or Perturbed_Nonbonded_Innerloop class
 * with the correct boundary conditions, virial and interaction term
 * enumeration values.
 */

/**
 * @define SPLIT_INTERACTION_FUNC
 * call a function f with a Interaction_Spec using the correct values for
 * boundary, virial and interaction term function to construct a
 * Nonbonded_Innerloop : split the interaction term function
 */
#define SPLIT_INTERACTION_FUNC(f, bound, ...) \
  switch(sim.param().force.interaction_function){ \
    case simulation::lj_crf_func : \
      f<Interaction_Spec<bound, simulation::lj_crf_func> >(__VA_ARGS__); \
      break; \
    case simulation::cgrain_func : \
      f<Interaction_Spec<bound, simulation::cgrain_func> >(__VA_ARGS__); \
      break; \
    default: \
      io::messages.add("wrong interaction function", "innerloop_template", io::message::error); \
      \
  } \


/**
 * @define SPLIT_INNERLOOP
 * call a function with a Interaction_Spec using the correct values for
 * boundary, virial and interaction term function for a
 * Nonbonded_Innerloop : split the boundary
 *
 */
#define SPLIT_INNERLOOP(f, ...) \
  switch(conf.boundary_type){ \
    case math::vacuum : \
      SPLIT_INTERACTION_FUNC(f, math::vacuum, __VA_ARGS__); \
      break; \
    case math::rectangular : \
      SPLIT_INTERACTION_FUNC(f, math::rectangular, __VA_ARGS__); \
      break; \
    case math::truncoct : \
      SPLIT_INTERACTION_FUNC(f, math::truncoct, __VA_ARGS__); \
      break; \
    default: \
      io::messages.add("wrong boundary type", "template_split", io::message::error); \
  } \

////////////////////////////////////////////////////////////////////////////////

/**
 * @define PERT_SPLIT_INTERACTION_FUNC
 * call a function f with a Interaction_Spec using the correct values for
 * boundary, virial and interaction term function to construct a
 * Nonbonded_Innerloop : split the interaction term function
 */
#define PERT_SPLIT_INTERACTION_FUNC(f, pertspec, bound, ...) \
  switch(sim.param().force.interaction_function){ \
    case simulation::lj_crf_func : \
      f< Interaction_Spec<bound, simulation::lj_crf_func>, \
         pertspec > (__VA_ARGS__); break; \
      break; \
    case simulation::cgrain_func : \
      f< Interaction_Spec<bound, simulation::cgrain_func>, \
         pertspec > (__VA_ARGS__); break; \
      break; \
    default: \
      io::messages.add("wrong interaction function", "innerloop_template", io::message::error); \
      \
  } \


/**
 * @define PERT_SPLIT_PERT_BOUNDARY
 * call a function with a Interaction_Spec using the correct values for
 * boundary, virial and interaction term function for a
 * Nonbonded_Innerloop : split the boundary
 */
#define PERT_SPLIT_BOUNDARY(f, pertspec, ...) \
  switch(conf.boundary_type){ \
    case math::vacuum : \
      PERT_SPLIT_INTERACTION_FUNC(f, pertspec, math::vacuum, __VA_ARGS__); \
      break; \
    case math::rectangular : \
      PERT_SPLIT_INTERACTION_FUNC(f, pertspec, math::rectangular, __VA_ARGS__); \
      break; \
    case math::truncoct : \
      PERT_SPLIT_INTERACTION_FUNC(f, pertspec, math::truncoct, __VA_ARGS__); \
      break; \
    default: \
      io::messages.add("wrong boundary type", "template_split", io::message::error); \
  } \

/**
 * @define SPLIT_PERT_INNERLOOP
 * call a function with a Interaction_Spec using the correct values for
 * boundary, virial and interaction term function for a
 * Nonbonded_Innerloop : split perturbation
 */
#define SPLIT_PERT_INNERLOOP(f, ...) \
  assert(sim.param().perturbation.perturbation); \
  if (sim.param().perturbation.scaling){ \
    PERT_SPLIT_BOUNDARY(f, Perturbation_Spec<scaling_on>, __VA_ARGS__); \
    } \
  else { \
    PERT_SPLIT_BOUNDARY(f, Perturbation_Spec<scaling_off>, __VA_ARGS__); \
  } \


////////////////////////////////////////////////////////////////////////////////

/**
 * @define SPLIT_PERTURBATION
 * call a function with the appropriate values for scaling
 * (assuming that perturbation is enabled).
 */
#define SPLIT_PERTURBATION(f, ...) \
assert(sim.param().perturbation.perturbation); \
if (sim.param().perturbation.scaling){ \
  f<Perturbation_Spec<scaling_on> >(__VA_ARGS__); \
  } else { \
  f<Perturbation_Spec<scaling_off> >(__VA_ARGS__); \
} \

////////////////////////////////////////////////////////////////////////////////


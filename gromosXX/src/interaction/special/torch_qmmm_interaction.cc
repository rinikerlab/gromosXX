/**
 * @file torch_interaction.cc
 * torch
 * 
 */

#include "../../stdheader.h"

#include "../../algorithm/algorithm.h"
#include "../../topology/topology.h"
#include "../../simulation/simulation.h"
#include "../../simulation/parameter.h"
#include "../../configuration/configuration.h"
#include "../../interaction/interaction.h"

#include "../../math/periodicity.h"
#include "../../math/gmath.h"
#include "../../interaction/interaction_types.h"

#include "../../interaction/special/torch_interaction.h"
#include "../../interaction/special/torch_qmmm_interaction.h"

#include "../../util/template_split.h"
#include "../../util/debug.h"

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

  Torch_QMMM_Interaction::Torch_QMMM_Interaction(const simulation::torch_model& model) : Torch_Interaction(model) { std::cout << "Torch: constructor: " << model.filename << std::endl; }

  Torch_QMMM_Interaction::~Torch_QMMM_Interaction() { std::cout << "Torch: destructor: " << model.filename << std::endl; }

  int Torch_QMMM_Interaction::init(topology::Topology & topo,
		                           configuration::Configuration & conf,
		                           simulation::Simulation & sim,
		                           std::ostream & os,
		                           bool quiet) {
    std::cout << "Torch: init" << std::endl;
    return 0;
  }

  int Torch_QMMM_Interaction::calculate_interactions(topology::Topology & topo,
				                                     configuration::Configuration & conf,
				                                     simulation::Simulation & sim) {
    std::cout << "Torch: calculate: " << model.filename << std::endl;
    return 0;
  }

  void Torch_QMMM_Interaction::forward() {

  }

  void Torch_QMMM_Interaction::backward() {

  }

}
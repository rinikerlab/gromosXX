/**
 * @file torch_interaction.cc
 * torch
 * 
 */

#include "../../stdheader.h"

#include "../../algorithm/algorithm.h"
#include "../../topology/topology.h"
#include "../../simulation/simulation.h"
#include "../../configuration/configuration.h"
#include "../../interaction/interaction.h"

#include "../../math/periodicity.h"
#include "../../math/gmath.h"
#include "../../interaction/interaction_types.h"

#include "../../interaction/special/torch_interaction.h"

#include "../../util/template_split.h"
#include "../../util/debug.h"

#include <torch/torch.h>
#include <torch/script.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

  int Torch_Interaction::init(topology::Topology & topo,
		                           configuration::Configuration & conf,
		                           simulation::Simulation & sim,
		                           std::ostream & os,
		                           bool quiet) {
    int err = load_model();
    if (err) return err;

    return err;
  }

  int Torch_Interaction::load_model() {
    int err = 0;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(model.filename);
    } catch (const c10::Error &e) {
      io::messages.add("Unable to load / deserialize Torch model: " + model.filename, io::message::error);
      err = 1;
      return err;
    }

    return err;
  }

  int Torch_Interaction::calculate_interactions(topology::Topology & topo,
				                                     configuration::Configuration & conf,
				                                     simulation::Simulation & sim) {
    int err = prepare_coordinates(sim);
    if (err) return err;

    err = prepare_tensors(sim);
    if (err) return err;

    err = forward();
    if (err) return err;

    err = backward();
    if (err) return err;

    err = update_energy();
    if (err) return err;

    err = update_forces();
    if (err) return err;

    return err;
  }

  int Torch_Interaction::forward() {
    std::cout << "Torch: Forward pass" << std::endl; 
    return 0; 
  }

  int Torch_Interaction::backward() {
    std::cout << "Torch: Backward pass" << std::endl;
    return 0; 
  }

  int Torch_Interaction::update_energy() { return 0; }

  int Torch_Interaction::update_forces() { return 0; }

} // namespace interaction
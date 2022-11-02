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
    DEBUG(15, "Initializing Torch interaction");
    int err = load_model();
    if (err) return err;

    return err;
  }

  int Torch_Interaction::load_model() {
    DEBUG(15, "Loading Torch model");
    int err = 0;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(model.filename);
    } catch (const c10::Error &e) {
      io::messages.add("Unable to load / deserialize Torch model: " + model.filename, io::message::error);
      err = 1;
      return err;
    }

    DEBUG(15, "Torch model successfully loaded");

    return err;
  }

  int Torch_Interaction::calculate_interactions(topology::Topology & topo,
				                                     configuration::Configuration & conf,
				                                     simulation::Simulation & sim) {
    DEBUG(15, "Calculating Torch Interaction");
    int err = prepare_input(sim);
    if (err) return err;

    err = build_tensors(sim);
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

} // namespace interaction
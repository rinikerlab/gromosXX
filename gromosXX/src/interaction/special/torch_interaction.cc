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

    return 0;
  }

  int Torch_Interaction::load_model() {
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(model.filename);
    } catch (const c10::Error &e) {
      io::messages.add("Unable to load / deserialize Torch model: " + model.filename, io::message::error);
      return 1;
    }

    return 0;
  }

} // namespace interaction
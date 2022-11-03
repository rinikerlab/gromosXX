/**
 * @file torch_interaction.cc
 * torch
 *
 */

#include "../../stdheader.h"

#include "../../algorithm/algorithm.h"
#include "../../configuration/configuration.h"
#include "../../interaction/interaction.h"
#include "../../simulation/simulation.h"
#include "../../topology/topology.h"

#include "../../interaction/interaction_types.h"
#include "../../math/gmath.h"
#include "../../math/periodicity.h"

#include "../../interaction/special/torch_interaction.h"

#include "../../util/debug.h"
#include "../../util/template_split.h"

#include <torch/script.h>
#include <torch/torch.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

int Torch_Interaction::init(topology::Topology &topo,
                            configuration::Configuration &conf,
                            simulation::Simulation &sim, std::ostream &os,
                            bool quiet) {
  DEBUG(15, "Initializing Torch interaction");
  m_timer.start(sim);
  m_timer.start_subtimer("Loading model");
  int err = load_model();
  if (err)
    return err;
  m_timer.stop_subtimer("Loading model");
  m_timer.stop();
  return err;
}

int Torch_Interaction::load_model() {
  DEBUG(15, "Loading Torch model");
  int err = 0;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model.filename);
  } catch (const c10::Error &e) {
    io::messages.add("Unable to load / deserialize Torch model: " +
                         model.filename,
                     io::message::error);
    err = 1;
    return err;
  }

  DEBUG(15, "Torch model successfully loaded");

  return err;
}

int Torch_Interaction::calculate_interactions(
    topology::Topology &topo, configuration::Configuration &conf,
    simulation::Simulation &sim) {
  DEBUG(15, "Calculating Torch Interaction");
  m_timer.start(sim);

  m_timer.start_subtimer("Preparing input");
  int err = prepare_input(sim);
  if (err)
    return err;
  m_timer.stop_subtimer("Preparing input");

  m_timer.start_subtimer("Building tensor");
  err = build_tensors(sim);
  if (err)
    return err;
  m_timer.stop_subtimer();

  m_timer.start_subtimer("Forward pass");
  err = forward();
  if (err)
    return err;
  m_timer.stop_subtimer("Forward pass");

  m_timer.start_subtimer("Backward pass");
  err = backward();
  if (err)
    return err;
  m_timer.stop_subtimer("Backward pass");

  m_timer.start_subtimer("Parsing tensors");
  err = get_energy();
  if (err)
    return err;
  err = get_forces();
  if (err)
    return err;
  m_timer.stop_subtimer("Parsing tensors");

  m_timer.start_subtimer("Writing data");
  err = write_data(topo, conf, sim);
  if (err)
    return err;
  m_timer.stop_subtimer("Writing data");

  m_timer.stop();

  return err;
}

} // namespace interaction
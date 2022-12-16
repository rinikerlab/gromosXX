
/**
 * @file in_torch.cc
 * implements torch methods.
 */

#include "../../stdheader.h"

#include "../../algorithm/algorithm.h"
#include "../../topology/topology.h"
#include "../../simulation/simulation.h"
#include "../../simulation/parameter.h"
#include "../../interaction/interaction_types.h"
#include "../../configuration/configuration.h"

#include "../../io/instream.h"
#include "../../io/blockinput.h"
#include "../../io/configuration/in_configuration.h"
#include "../../io/configuration/out_configuration.h"

#include "in_torch.h"

#include "../../util/debug.h"

#undef MODULE
#undef SUBMODULE
#define MODULE io
#define SUBMODULE topology

void io::In_Torch::read(topology::Topology &topo, simulation::Simulation &sim,
                        std::ostream &os) {
  io::messages.add("Reading Torch specification file", "In_Torch",
                   io::message::notice);

  read_models(sim);
}

void io::In_Torch::read_models(simulation::Simulation &sim) {
  std::vector<std::string> buffer = m_block["MODELS"];

  if (!buffer.size()) {
    io::messages.add(
        "TORCH requested but no models provided in Torch specification file.",
        "In_Torch", io::message::error);
    return;
  }
  _lineStream.clear();
  std::string bstr = concatenate(buffer.begin() + 1, buffer.end() - 1);
  // Strip away the last newline character
  bstr.pop_back();
  _lineStream.str(bstr);
  // block exists
  io::messages.add("Reading MODELS block in Torch specification file.",
                   "In_Torch", io::message::notice);
  while (!_lineStream.eof()) {
    simulation::torch_model model;
    unsigned atoms;
    int precision, device, write;
    _lineStream >> model.model_name >> atoms >> model.filename >> precision >> device >> write >> model.unit_factor_length >>
        model.unit_factor_energy >> model.unit_factor_force >> model.unit_factor_charge;
    if (_lineStream.fail()) {
      io::messages.add("Cannot read MODELS block", "In_Torch",
                       io::message::error);
      return;
    }

    // set enum for atom selection
    switch (atoms) {
    case 0:
      model.atom_selection = simulation::torch_all;
      break;
    case 1:
      model.atom_selection = simulation::torch_qmmm;
      break;
    case 2:
      model.atom_selection = simulation::torch_custom;
      break;
    default:
      io::messages.add(
          "Invalid atom selection specified in Torch specification file: " +
              std::to_string(atoms),
          "In_Torch", io::message::error);
      return;
    }

    // check what is supported
    if (model.atom_selection == simulation::torch_qmmm && sim.param().qmmm.qmmm != simulation::qmmm_electrostatic) {
      io::messages.add(
          "Can only combine Torch QM/MM Interface with electrostatic embedding scheme.",
          "In_Torch", io::message::error);
      return;
    }

    // set enum for model precision
    switch (precision) {
      case 0: model.precision = torch::kFloat16;
              break;
      case 1: model.precision = torch::kFloat32;
              break;
      case 2: model.precision = torch::kFloat64;
              break;
      default:
      io::messages.add(
          "Invalid precision specified in Torch specification file: " +
              precision,
          "In_Torch", io::message::error);
      return;
    }
    
    // set device
    switch (device) { // TODO: check ternary condition
      case 0: if (torch::cuda::is_available()) { torch::cuda::is_available() ? model.device = torch::kCUDA : model.device = torch::kCPU; } break;
      case 1: model.device = torch::kCPU; break;
      case 2: model.device = torch::kCUDA; break;
      default:
      io::messages.add(
          "Invalid device specified in Torch specification file: " +
              device,
          "In_Torch", io::message::error);
      return;
    }

    // NTWTORCH
    if (write < 0) {
      io::messages.add(
          "Invalid write-out frequency specified in Torch specification file: " +
              write,
          "In_Torch", io::message::error);
      return;
    }
    model.write = write;

    sim.param().torch.models.push_back(model);
  }
}
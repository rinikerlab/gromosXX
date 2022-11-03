
/**
 * @file in_torch.cc
 * implements torch methods.
 */


#include "../../stdheader.h"

#include "../../algorithm/algorithm.h"
#include "../../topology/topology.h"
#include "../../simulation/simulation.h"
#include "../../interaction/interaction_types.h"
#include "../../configuration/configuration.h"

#include "../../io/instream.h"
#include "../../io/blockinput.h"
#include "../../io/configuration/in_configuration.h"
#include "../../io/configuration/out_configuration.h"

#include <vector>

#include "in_torch.h"

#include "../../util/debug.h"

#undef MODULE
#undef SUBMODULE
#define MODULE io
#define SUBMODULE topology

void io::In_Torch::read(topology::Topology& topo,
        simulation::Simulation& sim,
        std::ostream & os) {
  io::messages.add("Reading Torch specification file",
                   "In_Torch", io::message::notice);
  
  read_models(sim);
}

void io::In_Torch::read_models(simulation::Simulation & sim)
{
  std::vector<std::string> buffer = m_block["MODELS"];

  if (!buffer.size()) {
    io::messages.add("TORCH requested but no models provided in Torch specification file.",
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
  unsigned atoms;
  std::string model_filename;
  double unit_factor_length, unit_factor_energy, unit_factor_force, unit_factor_charge;
  while(!_lineStream.eof()) {
    _lineStream >> atoms >> model_filename >> unit_factor_length >> unit_factor_energy >> unit_factor_force >> unit_factor_charge;
    if (_lineStream.fail()) {
      io::messages.add("Cannot read MODELS block", "In_Torch", io::message::error);
      return;
    }
    // set enum
    simulation::torch_atom_enum atom_selection;
    switch (atoms) {
      case 0: atom_selection = simulation::torch_all; break;
      case 1: atom_selection = simulation::torch_qmmm; break;
      case 2: atom_selection = simulation::torch_custom; break;
      default: io::messages.add("Invalid atom selection specified in Torch specification file: " + std::to_string(atoms), "In_Torch", io::message::error); return;
    }
    sim.param().torch.models.emplace_back(atom_selection, model_filename, unit_factor_length, unit_factor_energy, unit_factor_force, unit_factor_charge);
  }
}
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
#include <torch/torch.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

template <typename T>
int Torch_Interaction<T>::init(topology::Topology &topo,
                            configuration::Configuration &conf,
                            simulation::Simulation &sim, std::ostream &os,
                            bool quiet) {
  DEBUG(15, "Initializing Torch interaction");
  m_timer.start(sim);
  m_timer.start_subtimer("Loading model");
  int err = load_model();
  if (err)
    return err;

  if(!quiet) {
    os << "TORCH\n";
    os << "\tModel name: " << model.filename << '\n';
    os << "\tunits conversion factors: ";
    print_unit_factors(os);
    os << std::endl;
    os << "\tDevice: " << model.device << '\n';
    os << "\tPrecision: " << model.precision << "\n\n";
  }

  m_timer.stop_subtimer("Loading model");
  m_timer.stop();
  return err;
}

template <typename T>
int Torch_Interaction<T>::load_model() {
  DEBUG(15, "Loading Torch model");
  int err = 0;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model.filename, model.device);
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

template <typename T>
int Torch_Interaction<T>::calculate_interactions(
    topology::Topology &topo, configuration::Configuration &conf,
    simulation::Simulation &sim) {
  DEBUG(15, "Calculating Torch Interaction");
  m_timer.start(sim);

  m_timer.start_subtimer("Preparing input");
  int err = prepare_input(topo, conf, sim);
  if (err)
    return err;
  m_timer.stop_subtimer("Preparing input");

  if ((model.write > 0) &&
	  ((sim.steps()) % (model.write) == 0)) {
    m_timer.start_subtimer("Writing Torch input");
    // steps reported in output are steps finished already
    save_torch_input(sim.steps(), topo, conf, sim);
    m_timer.stop_subtimer("Writing Torch input");
  }

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
  err = update_energy(topo, conf, sim);
  if (err)
    return err;
  err = update_forces(topo, conf, sim);
  if (err)
    return err;
  m_timer.stop_subtimer("Parsing tensors");

  if ((model.write > 0) &&
	  ((sim.steps()) % (model.write) == 0)) {
    m_timer.start_subtimer("Writing Torch output");
    // steps reported in output are steps finished already
    save_torch_output(sim.steps(), topo, conf, sim);
    m_timer.stop_subtimer("Writing Torch output");
  }

  m_timer.stop();

  return err;
}

template <typename T>
void Torch_Interaction<T>::write_step_size(std::ofstream& ifs, 
                                           const unsigned int step) const {
  ifs << "TIMESTEP" << '\n';
  ifs << "    " << step << '\n';
  ifs << "END" << '\n';
}

template <typename T>
void Torch_Interaction<T>::write_coordinate_header(std::ofstream& ifs) const {
  // TURBOMOLE format
  ifs << "$coord" << '\n';
}

template <typename T>
void Torch_Interaction<T>::write_coordinate_footer(std::ofstream& ifs) const {
  // TURBOMOLE format
  ifs << "$end" << '\n';
}

template <typename T>
void Torch_Interaction<T>::write_gradient(const math::Vec& gradient, 
                                          std::ofstream& inputfile_stream) const {
  inputfile_stream.setf(std::ios::fixed, std::ios::floatfield);
  inputfile_stream << std::setprecision(12)
                   << std::setw(19) << gradient(0)
                   << std::setw(19) << gradient(1)
                   << std::setw(19) << gradient(2)
                   << '\n';
}

template <typename T>
void Torch_Interaction<T>::write_atom(std::ofstream& inputfile_stream
                                    , const int atomic_number
                                    , const math::Vec& pos) const {
  inputfile_stream.setf(std::ios::fixed, std::ios::floatfield);
  inputfile_stream << std::setprecision(20)
                   << std::setw(27) << pos(0)
                   << std::setw(27) << pos(1)
                   << std::setw(27) << pos(2)
                   << std::setw(8)  << atomic_number
                   << '\n';
}

template <typename T>
int Torch_Interaction<T>::open_input(std::ofstream& inputfile_stream, const std::string& input_file) const {
  inputfile_stream.open(input_file.c_str()); 
  if (!inputfile_stream.is_open()) {
    io::messages.add("Unable to write to file: "
            + input_file, this->model.model_name, io::message::error);
    return 1;
  }
  return 0;
}

// explicit instantiations: 
// https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl 
// https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file
template class Torch_Interaction<torch::Half>;
template class Torch_Interaction<float>;
template class Torch_Interaction<double>;

} // namespace interaction
/**
 * @file torch_global_interaction.cc
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
#include "../../interaction/special/torch_global_interaction.h"

#include "../../util/template_split.h"
#include "../../util/debug.h"

#include <torch/torch.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

template <typename T>
int Torch_Global_Interaction<T>::init(topology::Topology &topo,
                                 configuration::Configuration &conf,
                                 simulation::Simulation &sim, std::ostream &os,
                                 bool quiet) {
  DEBUG(15, "Initializing Torch Global Interaction");
  int err = Torch_Interaction<T>::init(topo, conf, sim, os, quiet);
  if (err)
    return err;

  // one batch per iteration
  batch_size = 1;
  dimensions = 3;

  // all atoms, natoms const over simulation
  natoms = topo.num_atoms();
  positions.resize(dimensions * natoms);

  // open trajectory streams
  if (this->model.write > 0) {
    // output filenames
    std::string trajectory_input_coordinate_file = this->model.model_name + ".coord";
    std::string trajectory_output_gradient_file = this->model.model_name + ".engrad";
    // coordinates
    int err = this->open_input(input_coordinate_stream, trajectory_input_coordinate_file);
    if (err) return err;
    // gradients
    err = this->open_input(output_gradient_stream, trajectory_output_gradient_file);
    if (err) return err;
  }

  // TODO: is PBC taken care of

  return err;
}

template <typename T>
int Torch_Global_Interaction<T>::prepare_input(const topology::Topology& topo, 
                                               const configuration::Configuration& conf, 
                                               const simulation::Simulation& sim) {
  DEBUG(15, "Preparing input");
  // Gromos -> Torch length unit is inverse of input value from Torch
  // specification file
  const double len_to_torch = 1.0 / this->model.unit_factor_length;
  int err = 0;

  DEBUG(15, "Transfering coordinates to Torch");
  for (unsigned idx = 0; idx < natoms; ++idx) {
    DEBUG(15, idx << " " << topo.qm_atomic_number(idx) << " "
                        << math::v2s(conf.current().pos(idx) * len_to_torch));
    math::vector_c2f<T>(positions, conf.current().pos(idx), idx, len_to_torch);
  }

  return err;
}

template <typename T>
int Torch_Global_Interaction<T>::build_tensors(const simulation::Simulation &sim) {
  DEBUG(15, "Building tensors");
  int err = 0;
  // batch size is 1
  positions_tensor = torch::from_blob(positions.data(), {batch_size, natoms, dimensions},
                       this->tensor_float_gradient);
  return err;
}

template <typename T>
int Torch_Global_Interaction<T>::forward() {
  DEBUG(15, "Calling forward on the model");
  int err = 0;
  torch::jit::IValue input = positions_tensor;
  energy_tensor = this->module.forward({input}).toTensor();
  return err;
}

template <typename T>
int Torch_Global_Interaction<T>::backward() {
  DEBUG(15, "Calling backward on the model");
  int err = 0;
  energy_tensor.backward();
  gradient_tensor = positions_tensor.grad();
  DEBUG(15, "Norm of gradient tensor: " +
                std::to_string(torch::sum(gradient_tensor).item<T>()));
  return err;
}

template <typename T>
int Torch_Global_Interaction<T>::update_energy(topology::Topology &topo,
                                             configuration::Configuration &conf,
                                             const simulation::Simulation &sim) {
  double energy = static_cast<double>(energy_tensor.item<T>()) *
                  this->model.unit_factor_energy;
  DEBUG(15, "Parsing Torch energy: " << energy << " kJ / mol");
  // TODO: temporary storage for trajectory write-out
  conf.current().energies.torch_total = energy;
  // TODO: split energies depending on which model it is and print
  return 0;
}

template <typename T>
int Torch_Global_Interaction<T>::update_forces(topology::Topology &topo,
                                               configuration::Configuration &conf,
                                               const simulation::Simulation &sim) {
  DEBUG(15, "Parsing Torch forces");

  // calculate virial and update forces
  math::Matrix virial_tensor(0.0);

  for (unsigned idx = 0; idx < natoms; ++idx) {
    DEBUG(15, "Parsing gradients of atom " << i);
    math::Vec force;
    // forces = negative gradient (!)
    force(0) = -1.0 * static_cast<double>(gradient_tensor[batch_size - 1][idx][0].item<T>()) *  // 0 idx for batch_size
          this->model.unit_factor_force;
    force(1) = -1.0 * static_cast<double>(gradient_tensor[batch_size - 1][idx][1].item<T>()) *
          this->model.unit_factor_force;
    force(2) = -1.0 * static_cast<double>(gradient_tensor[batch_size - 1][idx][2].item<T>()) *
          this->model.unit_factor_force;     
    DEBUG(15, "Atom " << idx << ", force: " << math::v2s(force));
    conf.current().force(idx) += force; 
    // virial calculation
    math::Vec& pos = conf.current().pos(idx);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        virial_tensor(j,i) += pos(j) * force(i);
      }
    }
  }
  
  // add virial contribution
  if (sim.param().pcouple.virial) {
    conf.current().virial_tensor += virial_tensor;
  }

  return 0;
}

template <typename T>
void Torch_Global_Interaction<T>::save_torch_input(const unsigned int step
                                                 , const topology::Topology& topo
                                                 , const configuration::Configuration& conf
                                                 , const simulation::Simulation& sim) {
  save_input_coord(input_coordinate_stream, step, topo, conf, sim);
}

template <typename T>
void Torch_Global_Interaction<T>::save_torch_output(const unsigned int step
                                                  , const topology::Topology& topo
                                                  , const configuration::Configuration& conf
                                                  , const simulation::Simulation& sim) {
  save_output_gradients(output_gradient_stream, step, topo, conf, sim);
}

template <typename T>
void Torch_Global_Interaction<T>::save_input_coord(std::ofstream& ifs
                                                 , const unsigned int step
                                                 , const topology::Topology& topo
                                                 , const configuration::Configuration& conf
                                                 , const simulation::Simulation& sim) {
  // Gromos -> Bohr (Turbomole format)
  const double len_to_qm = 1.0 / 0.05291772109;

  // write step size
  this->write_step_size(ifs, step);

  // write coordinates
  this->write_coordinate_header(ifs);
  DEBUG(15, "Writing Torch coordinates");
  for (unsigned idx = 0; idx < natoms; ++idx) {
    DEBUG(15, idx << " " << topo.qm_atomic_number(idx) << " " << math::v2s(conf.current().pos(idx) * len_to_qm));
    this->write_atom(ifs, topo.qm_atomic_number(idx), conf.current().pos(idx) * len_to_qm);
  }
  
  this->write_coordinate_footer(ifs);
}

template <typename T>
void Torch_Global_Interaction<T>::save_output_gradients(std::ofstream& ifs
                                                      , const unsigned int step
                                                      , const topology::Topology& topo
                                                      , const configuration::Configuration& conf
                                                      , const simulation::Simulation& sim) {
  // Gromos -> Hartree / Bohr (Turbomole format)
  const double energy_to_qm = 1.0 / 2625.499639;
  const double force_to_qm = 1.0 / (2625.499639 / 0.05291772109);

  // write step size
  this->write_step_size(ifs, step);

  // Write energy
  ifs.setf(std::ios::fixed, std::ios::floatfield);
  ifs << std::setprecision(12);
  double energy = static_cast<double>(energy_tensor.item<T>()) *
                  this->model.unit_factor_energy;
  ifs << "ENERGY: " << energy * energy_to_qm << '\n';

  // write forces
  DEBUG(15, "Writing Torch gradients");
  for (unsigned idx = 0; idx < natoms; ++idx) {
    math::Vec force;
    // forces = negative gradient (!)
    force(0) = -1.0 * static_cast<double>(gradient_tensor[batch_size - 1][idx][0].item<T>()) *  // 0 idx for batch_size
          this->model.unit_factor_force;
    force(1) = -1.0 * static_cast<double>(gradient_tensor[batch_size - 1][idx][1].item<T>()) *
          this->model.unit_factor_force;
    force(2) = -1.0 * static_cast<double>(gradient_tensor[batch_size - 1][idx][2].item<T>()) *
          this->model.unit_factor_force;     
    DEBUG(15, "Atom " << idx << ", force: " << math::v2s(force));
    // forces = negative gradient (!)
    DEBUG(15, "Writing gradients of atom " << idx);
    this->write_gradient(-1.0 * force * force_to_qm, ifs);
    DEBUG(15, "Force: " << math::v2s(force));
  }
  
}

// explicit instantiations
template class Torch_Global_Interaction<torch::Half>;
template class Torch_Global_Interaction<float>;
template class Torch_Global_Interaction<double>;

} // namespace interaction
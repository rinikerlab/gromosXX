/**
 * @file torch_qmmm_interaction.cc
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

#include "../../interaction/qmmm/qmmm_interaction.h"
#include "../../interaction/qmmm/qm_zone.h"
#include "../../interaction/qmmm/qm_atom.h"
#include "../../interaction/qmmm/qm_link.h"
#include "../../interaction/qmmm/mm_atom.h"

#include "../../interaction/special/torch_interaction.h"
#include "../../interaction/special/torch_qmmm_interaction.h"

#include "../../util/template_split.h"
#include "../../util/debug.h"

#include <cassert>
#include <tuple>

#include <torch/torch.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

template <typename T>
int Torch_QMMM_Interaction<T>::init(topology::Topology &topo,
                                 configuration::Configuration &conf,
                                 simulation::Simulation &sim, std::ostream &os,
                                 bool quiet) {
  DEBUG(15, "Initializing Torch QM/MM Interaction");
  int err = Torch_Interaction<T>::init(topo, conf, sim, os, quiet);
  if (err)
    return err;

  // one batch per iteration
  batch_size = 1;
  dimensions = 3;

  // TODO: copy QM zone instead of using pointers, would make virial calculation easier
  // TODO: is PBC taken care of
  // TODO: get_num_of charges from QM_Worker

  err = init_qm_zone();
  if (err)
    return err;

  err = init_qm_atom_numbers();
  if (err)
    return err;

  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::init_qm_zone() {
  int err = 0;
  qmmm_ptr = QMMM_Interaction::pointer();
  if (qmmm_ptr == nullptr) {
    io::messages.add(
        "Unable to get QMMM interaction in Torch QM/MM Interaction set-up",
        "Torch_QMMM_Interaction", io::message::error);
    err = 1;
    return err;
  }
  if (qmmm_ptr->qm_zone() == nullptr) {
    io::messages.add("Unable to get QM zone in Torch QM/MM Interaction set-up",
                     "Torch_QMMM_Interaction", io::message::error);
    err = 1;
    return err;
  }

  // initial copy of the QM zone
  qm_zone = *(qmmm_ptr->qm_zone());

  // number of atoms in QM zone should not change over simulation
  natoms = qm_zone.qm.size() + qm_zone.link.size();

  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::init_qm_atom_numbers() {
  int err = 0;
  qm_atomic_numbers.resize(natoms);
  qm_positions.resize(dimensions * natoms);

  unsigned int i = 0;
  DEBUG(15, "Initializing QM atom types");
  for (std::set<QM_Atom>::const_iterator it = qm_zone.qm.begin(),
                                         to = qm_zone.qm.end();
       it != to; ++it, ++i) {
    DEBUG(15, it->index << " " << it->atomic_number);
    this->qm_atomic_numbers[i] = it->atomic_number;
  }
  // QM link atoms (iterator i keeps running)
  DEBUG(15, "Initializing capping atom types");
  for (std::set<QM_Link>::const_iterator it = qm_zone.link.begin(),
                                         to = qm_zone.link.end();
       it != to; ++it, ++i) {
    DEBUG(15, "Capping atom " << it->qm_index << "-" << it->mm_index << " "
                              << it->atomic_number);
    this->qm_atomic_numbers[i] = it->atomic_number;
  }
  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::prepare_input(const simulation::Simulation &sim) {
  // get a new copy of the QM zone
  qm_zone = *(qmmm_ptr->qm_zone());

  // get size of QM zone and MM zone
  assert(static_cast<int>(qm_zone.qm.size() + qm_zone.link.size()) ==
         natoms);
  ncharges = get_num_charges(sim);

  // put coordinates into one-dimensional vectors
  int err = prepare_qm_atoms();
  if (err)
    return err;

  err = prepare_mm_atoms();
  if (err)
    return err;

  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::prepare_qm_atoms() {
  int err = 0;
  // Gromos -> Torch length unit is inverse of input value from Torch
  // specification file
  const double len_to_torch = 1.0 / this->model.unit_factor_length;

  // transfer QM coordinates
  DEBUG(15, "Transfering QM coordinates to Torch");
  unsigned int i = 0;
  for (std::set<QM_Atom>::const_iterator it = qm_zone.qm.begin(),
                                         to = qm_zone.qm.end();
       it != to; ++it) {
    DEBUG(15, it->index << " " << it->atomic_number << " "
                        << math::v2s(it->pos * len_to_torch));
    math::vector_c2f<T>(qm_positions, it->pos, i, len_to_torch);
    ++i;
  }
  // transfer capping atoms (index i keeps running...)
  DEBUG(15, "Transfering capping atoms coordinates to Torch");
  for (std::set<QM_Link>::const_iterator it = qm_zone.link.begin(),
                                         to = qm_zone.link.end();
       it != to; it++) {
    DEBUG(15, "Capping atom " << it->qm_index << "-" << it->mm_index << " "
                              << it->atomic_number << " "
                              << math::v2s(it->pos * len_to_torch));
    math::vector_c2f<T>(qm_positions, it->pos, i, len_to_torch);
    ++i;
  }
  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::prepare_mm_atoms() {
  int err = 0;
  // Gromos -> Torch length unit is inverse of input value from Torch
  // specification file
  const double len_to_torch = 1.0 / this->model.unit_factor_length;
  const double cha_to_torch = 1.0 / this->model.unit_factor_charge;

  mm_atomic_numbers.resize(ncharges);
  mm_charges.resize(ncharges);
  mm_positions.resize(ncharges * dimensions);

  DEBUG(15, "Transfering point charges to Torch");

  unsigned int i =
      0; // iterate over atoms - keep track of the offset for Fortran arrays
  for (std::set<MM_Atom>::const_iterator it = qm_zone.mm.begin(),
                                         to = qm_zone.mm.end();
       it != to; ++it) {
    // memory layout of point charge arrays:
    // numbers (1d), charges (1d), point_charges (3d): one-dimensional arrays
    // COS numbers, charges, cartesian coordinates are after MM numbers,
    // charges, cartesian coordinates
    if (it->is_polarisable) {
      // MM atom minus COS
      DEBUG(15, it->index << " " << it->atomic_number << " "
                          << (it->charge - it->cos_charge) * cha_to_torch << " "
                          << math::v2s(it->pos * len_to_torch));
      mm_atomic_numbers[i] = it->atomic_number;
      mm_charges[i] = (it->charge - it->cos_charge) * cha_to_torch;
      math::vector_c2f<T>(mm_positions, it->pos, i, len_to_torch);
      ++i;
      // COS
      DEBUG(15, it->index << " " << it->atomic_number << " " << it->cos_charge
                          << " "
                          << math::v2s((it->pos + it->cosV) * cha_to_torch));
      mm_atomic_numbers[i] = it->atomic_number;
      mm_charges[i] = it->cos_charge * cha_to_torch;
      math::vector_c2f<T>(mm_positions, it->cosV, i, len_to_torch);
      ++i;
    } else {
      DEBUG(15, it->index << " " << it->atomic_number << " "
                          << it->charge * cha_to_torch << " "
                          << math::v2s(it->pos * len_to_torch));
      mm_atomic_numbers[i] = it->atomic_number;
      mm_charges[i] = it->charge * cha_to_torch;
      math::vector_c2f<T>(mm_positions, it->pos, i, len_to_torch);
      ++i;
    }
  }
  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::build_tensors(const simulation::Simulation &sim) {
  DEBUG(15, "Building tensors");
  int err = 0;
  // batch size is 1
  qm_atomic_numbers_tensor = torch::from_blob(
      qm_atomic_numbers.data(), {batch_size, natoms}, this->tensor_int64);
  qm_positions_tensor =
      torch::from_blob(qm_positions.data(), {batch_size, natoms, dimensions},
                       this->tensor_float_gradient);
  mm_atomic_numbers_tensor = torch::from_blob(
      mm_atomic_numbers.data(), {batch_size, ncharges}, this->tensor_int64);
  mm_charges_tensor =
      torch::from_blob(mm_charges.data(), {batch_size, ncharges},
                       this->tensor_float_no_gradient);
  mm_positions_tensor =
      torch::from_blob(mm_positions.data(), {batch_size, ncharges, dimensions},
                       this->tensor_float_gradient);

  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::forward() {
  DEBUG(15, "Calling forward on the model");
  int err = 0;
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
      input_tuple(qm_atomic_numbers_tensor, qm_positions_tensor,
                  mm_atomic_numbers_tensor, mm_charges_tensor,
                  mm_positions_tensor);
  energy_tensor = this->module.forward({input_tuple}).toTensor();
  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::backward() {
  DEBUG(15, "Calling backward on the model");
  int err = 0;
  energy_tensor.backward();
  qm_gradient_tensor = qm_positions_tensor.grad();
  mm_gradient_tensor = mm_positions_tensor.grad();
  DEBUG(15, "Sum of QM gradient tensor: " +
                std::to_string(torch::sum(qm_gradient_tensor).item<T>()));
  DEBUG(15, "Sum of MM gradient tensor: " +
                std::to_string(torch::sum(mm_gradient_tensor).item<T>()));
  return err;
}

template <typename T>
int Torch_QMMM_Interaction<T>::update_energy(topology::Topology &topo,
                                             configuration::Configuration &conf,
                                             const simulation::Simulation &sim) {
  double energy = static_cast<double>(energy_tensor.item<T>()) *
                  this->model.unit_factor_energy;
  DEBUG(15, "Parsing Torch energy: " << energy << " kJ / mol");
  conf.current().energies.torch_total = energy;
  // TODO: split energies depending on which model it is and print
  return 0;
}

template <typename T>
int Torch_QMMM_Interaction<T>::update_forces(topology::Topology &topo,
                                             configuration::Configuration &conf,
                                             const simulation::Simulation &sim) {
  // QM atoms and QM links
  unsigned qm_atom = 0;

  // Parse forces on QM atoms 
  for (std::set<QM_Atom>::iterator it = qm_zone.qm.begin(),
                                   to = qm_zone.qm.end();
       it != to; ++it) {
    DEBUG(15, "Parsing gradients of QM atom " << it->index);
    math::Vec force;
    // forces = negative gradient (!)
    force[0] = -1.0 * static_cast<double>(qm_gradient_tensor[batch_size - 1][qm_atom][0].item<T>()) *  // 0 idx for batch_size
          this->model.unit_factor_force;
    force[1] = -1.0 * static_cast<double>(qm_gradient_tensor[batch_size - 1][qm_atom][1].item<T>()) *
          this->model.unit_factor_force;
    force[2] = -1.0 * static_cast<double>(qm_gradient_tensor[batch_size - 1][qm_atom][2].item<T>()) *
          this->model.unit_factor_force;     
    DEBUG(15, "Force: " << math::v2s(force));
    it->force = force; // save copy for virial calculation
    ++qm_atom;
  }

  // Parse forces on capping atoms (index i keeps running...) 
  for (std::set<QM_Link>::iterator it = qm_zone.link.begin(),
                                   to = qm_zone.link.end();
       it != to; ++it) {
    DEBUG(15, "Parsing gradient of capping atom " << it->qm_index << "-"
                                                  << it->mm_index);
    math::Vec force;
    force[0] = -1.0 * static_cast<double>(qm_gradient_tensor[batch_size - 1][qm_atom][0].item<T>()) *  // 0 idx for batch_size
          this->model.unit_factor_force;
    force[1] = -1.0 * static_cast<double>(qm_gradient_tensor[batch_size - 1][qm_atom][1].item<T>()) *
          this->model.unit_factor_force;
    force[2] = -1.0 * static_cast<double>(qm_gradient_tensor[batch_size - 1][qm_atom][2].item<T>()) *
          this->model.unit_factor_force;
    it->force = force; // save copy for distribution later
    DEBUG(15, "Redistributing force of capping atom");
    DEBUG(15, "QM-MM link: " << it->qm_index << " - " << it->mm_index);
    std::set<QM_Atom>::iterator qm_it = this->qm_zone.qm.find(it->qm_index);
    std::set<MM_Atom>::iterator mm_it = this->qm_zone.mm.find(it->mm_index);
    assert(qm_it != this->qm_zone.qm.end());
    assert(mm_it != this->qm_zone.mm.end());
    // distribute forces over QM and MM (do before virial calculation)
    it->distribute_force(qm_it->pos, mm_it->pos, qm_it->force, mm_it->force);
    ++qm_atom;
  }

  // Parse forces on MM atoms
  unsigned int mm_atom = 0;
  for (std::set<MM_Atom>::iterator it = qm_zone.mm.begin(),
                                   to = qm_zone.mm.end();
       it != to; ++it) {
    DEBUG(15, "Parsing gradient of MM atom " << it->index);
    math::Vec force;
    // forces = negative gradient (!)
    force[0] = -1.0 * static_cast<double>(mm_gradient_tensor[batch_size - 1][mm_atom][0].item<T>()) *  // 0 idx for batch_size
          this->model.unit_factor_force;
    force[1] = -1.0 * static_cast<double>(mm_gradient_tensor[batch_size - 1][mm_atom][1].item<T>()) *
          this->model.unit_factor_force;
    force[2] = -1.0 * static_cast<double>(mm_gradient_tensor[batch_size - 1][mm_atom][2].item<T>()) *
          this->model.unit_factor_force;
    DEBUG(15, "Force: " << math::v2s(force));
    it->force = force; // save copy for virial calculation
    if (it->is_polarisable) {
      ++mm_atom; // COS gradients live directly past the corresponding MM
                 // gradients
      DEBUG(15, "Parsing gradient of COS of MM atom " << it->index);
      math::Vec cos_force;
      cos_force[0] = -1.0 * static_cast<double>(mm_gradient_tensor[batch_size - 1][mm_atom][0].item<T>()) * // 0 idx for batch_size
            this->model.unit_factor_force;
      cos_force[1] = -1.0 * static_cast<double>(mm_gradient_tensor[batch_size - 1][mm_atom][1].item<T>()) *
            this->model.unit_factor_force;
      cos_force[2] = -1.0 * static_cast<double>(mm_gradient_tensor[batch_size - 1][mm_atom][2].item<T>()) *
            this->model.unit_factor_force;
      DEBUG(15, "Force " << math::v2s(cos_force));
      it->cos_force = cos_force; // save copy for virial calculation
    }
    ++mm_atom;
  }

  // calculate virial and update forces
  // TODO: COS not processed / updated
  math::Matrix virial_tensor(0.0);

  // QM atoms
  for (std::set<QM_Atom>::const_iterator
      it = qm_zone.qm.begin(), to = qm_zone.qm.end(); it != to; ++it)
    {
    DEBUG(15, "Atom " << it->index << ", force: " << math::v2s(it->force));
    conf.current().force(it->index) += it->force;
    math::Vec& pos = conf.current().pos(it->index);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        virial_tensor(j,i) += pos(j) * it->force(i);
      }
    }
  }

  // MM atoms
  for (std::set<MM_Atom>::const_iterator
      it = qm_zone.mm.begin(), to = qm_zone.mm.end(); it != to; ++it)
    {
    DEBUG(15, "Atom " << it->index << ", force: " << math::v2s(it->force));
    conf.current().force(it->index) += it->force;
    math::Vec& pos = conf.current().pos(it->index);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        virial_tensor(j,i) += pos(j) * it->force(i);
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
int Torch_QMMM_Interaction<T>::get_num_charges(const simulation::Simulation &sim) const {
  int num_charges = 0;
  switch (sim.param().qmmm.qmmm) {
  case simulation::qmmm_mechanical: {
    num_charges = 0;
    break;
  }
  case simulation::qmmm_electrostatic: {
    num_charges = qm_zone.mm.size();
    break;
  }
  case simulation::qmmm_polarisable: {
    num_charges = qm_zone.mm.size();
    for (std::set<MM_Atom>::const_iterator it = qm_zone.mm.begin(),
                                           to = qm_zone.mm.end();
         it != to; ++it) {
      num_charges += int(it->is_polarisable);
    }
    break;
  }
  default: {
    io::messages.add("Unknown QMMM option", this->name, io::message::error);
  }
  }
  return num_charges;
}

// explicit instantiations
template class Torch_QMMM_Interaction<torch::Half>;
template class Torch_QMMM_Interaction<float>;
template class Torch_QMMM_Interaction<double>;

} // namespace interaction
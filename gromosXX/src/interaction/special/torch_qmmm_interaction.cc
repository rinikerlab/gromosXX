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

#include "../../interaction/qmmm/qmmm_interaction.h"
#include "../../interaction/qmmm/qm_zone.h"
#include "../../interaction/qmmm/qm_atom.h"
#include "../../interaction/qmmm/qm_link.h"
#include "../../interaction/qmmm/mm_atom.h"

#include "../../interaction/special/torch_interaction.h"
#include "../../interaction/special/torch_qmmm_interaction.h"

#include "../../util/template_split.h"
#include "../../util/debug.h"

#include <tuple>

#include <torch/torch.h>

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special

namespace interaction {

  Torch_QMMM_Interaction::Torch_QMMM_Interaction(const simulation::torch_model& model) : Torch_Interaction(model) {}

  Torch_QMMM_Interaction::~Torch_QMMM_Interaction() = default;

  int Torch_QMMM_Interaction::init(topology::Topology & topo,
		                           configuration::Configuration & conf,
		                           simulation::Simulation & sim,
		                           std::ostream & os,
		                           bool quiet) {
    int err = Torch_Interaction::init(topo, conf, sim, os, quiet);
    if (err) return err;

    err = init_qm_zone();
    if (err) return err;

    err = init_qm_atom_numbers();
    if (err) return err;

    return err;
  }

  int Torch_QMMM_Interaction::init_qm_zone() {
    int err = 0;
    QMMM_Interaction* qmmm = QMMM_Interaction::pointer();
    if (qmmm == nullptr) {
      io::messages.add("Unable to get QMMM interaction in Torch QM/MM Interaction set-up",
                        "Torch_QMMM_Interaction", io::message::error);
      err = 1;
      return err;
    }
    qm_zone_ptr = qmmm->qm_zone();
    if (qm_zone_ptr == nullptr) {
      io::messages.add("Unable to get QM zone in Torch QM/MM Interaction set-up",
                        "Torch_QMMM_Interaction", io::message::error);
      err = 1;
      return err;
    }

    natoms = qm_zone_ptr->qm.size() + qm_zone_ptr->link.size();

    return err;
  }

  int Torch_QMMM_Interaction::prepare_input(const simulation::Simulation& sim) {
    // get size of QM zone and MM zone
    ncharges = get_num_charges(sim);

    // put coordinates into one-dimensional vectors
    int err = prepare_qm_atoms();
    if (err) return err;
    
    err = prepare_mm_atoms();
    if (err) return err;

    return err;
  }

  int Torch_QMMM_Interaction::init_qm_atom_numbers() {
    int err = 0;
    qm_atomic_numbers.resize(natoms);
    qm_positions.resize(3 * natoms);

    unsigned int i = 0;
    DEBUG(15, "Initializing QM atom types");
    for (std::set<QM_Atom>::const_iterator 
         it = qm_zone_ptr->qm.begin(), to = qm_zone_ptr->qm.end(); it != to; ++it, ++i) {
    DEBUG(15, it->index << " " << it->atomic_number);
    this->qm_atomic_numbers[i] = it->atomic_number;
    }
    // QM link atoms (iterator i keeps running)
    DEBUG(15, "Initializing capping atom types");
    for (std::set<QM_Link>::const_iterator
           it = qm_zone_ptr->link.begin(), to = qm_zone_ptr->link.end(); it != to; ++it, ++i) {
      DEBUG(15, "Capping atom " << it->qm_index << "-" << it->mm_index << " "
        << it->atomic_number);
      this->qm_atomic_numbers[i] = it->atomic_number;
    }
    return err;
  }

  int Torch_QMMM_Interaction::prepare_qm_atoms() {
    int err = 0;

    // transfer QM coordinates
    DEBUG(15, "Transfering QM coordinates to Torch");
    unsigned int i = 0;
    for (std::set<QM_Atom>::const_iterator 
           it = qm_zone_ptr->qm.begin(), to = qm_zone_ptr->qm.end(); it != to; ++it) {
      DEBUG(15, it->index << " " << it->atomic_number << " " << math::v2s(it->pos));
      math::vector_c2f<float>(qm_positions, it->pos, i);
      ++i;
    }
    // transfer capping atoms (index i keeps running...)
    DEBUG(15, "Transfering capping atoms coordinates to Torch");
    for (std::set<QM_Link>::const_iterator it = qm_zone_ptr->link.begin(), to = qm_zone_ptr->link.end(); it != to; it++) {
      DEBUG(15, "Capping atom " << it->qm_index << "-" << it->mm_index << " "
        << it->atomic_number << " " << math::v2s(it->pos));
      math::vector_c2f<float>(qm_positions, it->pos, i);
      ++i;
    } 
    return err;
  }

  int Torch_QMMM_Interaction::prepare_mm_atoms() {
    int err = 0;

   mm_atomic_numbers.resize(ncharges);
   mm_charges.resize(ncharges);
   mm_positions.resize(ncharges * 3);

  DEBUG(15, "Transfering point charges to Torch");

  unsigned int i = 0; // iterate over atoms - keep track of the offset for Fortran arrays
  for (std::set<MM_Atom>::const_iterator
         it = qm_zone_ptr->mm.begin(), to = qm_zone_ptr->mm.end(); it != to; ++it) {
    // memory layout of point charge arrays: 
    // numbers (1d), charges (1d), point_charges (3d): one-dimensional arrays 
    // COS numbers, charges, cartesian coordinates are after MM numbers, charges, cartesian coordinates
    if (it->is_polarisable) {
      // MM atom minus COS
      DEBUG(15, it->index << " " << it->atomic_number << " " 
        << (it->charge - it->cos_charge) << " " << math::v2s(it->pos));
      mm_atomic_numbers[i] = it->atomic_number;
      mm_charges[i] = (it->charge - it->cos_charge);
      math::vector_c2f<float>(mm_positions, it->pos, i);
      ++i;
      // COS
      DEBUG(15, it->index << " " << it->atomic_number << " " 
        << it->cos_charge << " " << math::v2s((it->pos + it->cosV)));
      mm_atomic_numbers[i] = it->atomic_number;
      mm_charges[i] = it->cos_charge;
      math::vector_c2f<float>(mm_positions, it->cosV, i);
      ++i;
    }
    else {
      DEBUG(15, it->index << " " << it->atomic_number << " " 
        << it->charge << " " << math::v2s(it->pos));
      mm_atomic_numbers[i] = it->atomic_number;
      mm_charges[i] = it->charge;
      math::vector_c2f<float>(mm_positions, it->pos, i);
      ++i;
    }
  }
    return err;
  }

  int Torch_QMMM_Interaction::build_tensors(const simulation::Simulation& sim) {
    int err = 0;
    qm_atomic_numbers_tensor = torch::from_blob(qm_atomic_numbers.data(), {natoms}, sim.param().torch.options_int);
    qm_positions_tensor      = torch::from_blob(qm_positions.data(), {natoms, 3}, sim.param().torch.options_float_gradient);
    mm_atomic_numbers_tensor = torch::from_blob(mm_atomic_numbers.data(), {ncharges}, sim.param().torch.options_int);
    mm_charges_tensor        = torch::from_blob(mm_charges.data(), {ncharges}, sim.param().torch.options_float_no_gradient);
    mm_positions_tensor      = torch::from_blob(mm_positions.data(), {ncharges, 3}, sim.param().torch.options_float_gradient);

    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "torch::Tensor from std::vector<float> QM:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Atomic numbers QM:" << std::endl;
    std::cout << qm_atomic_numbers_tensor << std::endl;
    std::cout << "Atomic positions QM:" << std::endl;
    std::cout << qm_positions_tensor << std::endl;

    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "torch::Tensor from std::vector<float> MM:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Atomic numbers MM:" << std::endl;
    std::cout << mm_atomic_numbers_tensor << std::endl;
    std::cout << "Atomic charges MM:" << std::endl;
    std::cout << mm_charges_tensor << std::endl;
    std::cout << "Atomic positions MM:" << std::endl;
    std::cout << mm_positions_tensor << std::endl;

    return err;
  }

  int Torch_QMMM_Interaction::forward() {
    int err = 0;
    std::tuple<torch::Tensor,torch::Tensor> input_tuple(qm_positions_tensor, qm_positions_tensor);
    energy_tensor = module.forward({input_tuple}).toTensor();
    std::cout << "Energy tensor: " << std::endl;
    std::cout << energy_tensor << std::endl;
    return err;
  }

  int Torch_QMMM_Interaction::backward() {
    int err = 0;
    energy_tensor.backward();
    qm_gradient_tensor = qm_positions_tensor.grad();
    mm_gradient_tensor = mm_positions_tensor.grad();
    std::cout << "QM gradient tensor: " << std::endl;
    std::cout << qm_gradient_tensor << std::endl;
    std::cout << "MM gradient tensor: " << std::endl;
    std::cout << mm_gradient_tensor << std::endl;
    return err;
  }

  int Torch_QMMM_Interaction::get_num_charges(const simulation::Simulation& sim) {
  int num_charges = 0;
  switch (sim.param().qmmm.qmmm) {
    case simulation::qmmm_mechanical: {
      num_charges = 0;
      break;
    }
    case simulation::qmmm_electrostatic: {
      num_charges = qm_zone_ptr->mm.size();
      break;
    }
    case simulation::qmmm_polarisable: {
      num_charges = qm_zone_ptr->mm.size();
      for (std::set<MM_Atom>::const_iterator
          it = qm_zone_ptr->mm.begin(), to = qm_zone_ptr->mm.end(); it != to; ++it) {
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

}
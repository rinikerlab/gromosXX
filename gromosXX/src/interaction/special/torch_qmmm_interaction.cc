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

    err = initialize_qm_zone();
    if (err) return err;

    err = initialize_qm_atom_types();
    if (err) return err;

    return err;
  }

  int Torch_QMMM_Interaction::initialize_qm_zone() {
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
    return err;
  }

  int Torch_QMMM_Interaction::initialize_qm_atom_types() {
    int err = 0;
    qm_atomic_numbers.resize(qm_zone_ptr->qm.size());

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

  int Torch_QMMM_Interaction::prepare_coordinates(simulation::Simulation& sim) {
    int err = 0;
    return err;
  }

  int Torch_QMMM_Interaction::prepare_tensors(simulation::Simulation& sim) {
    int err = 0;
    qm_atomic_numbers_tensor = torch::from_blob(qm_atomic_numbers.data(), {qm_zone_ptr->qm.size()}, sim.param().torch.options_no_gradient);

    return err;
  }

}
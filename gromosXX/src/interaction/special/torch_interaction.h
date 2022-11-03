/**
 * @file torch_interaction.h
 * torch
 */

#ifndef INCLUDED_TORCH_INTERACTION_H
#define INCLUDED_TORCH_INTERACTION_H

#include <vector>

#include <torch/torch.h>

#include "../../stdheader.h"
#include "../../interaction/interaction.h"
#include "../../simulation/parameter.h"

namespace interaction {

/*
 * Provides an interface to PyTorch
 */
class Torch_Interaction : public Interaction {

public:
  /**
   * Constructor - Initializes name and timer of the Interaction
   *
   */
  Torch_Interaction(const simulation::torch_model &model,
                    const std::string &name)
      : Interaction(name), model(model) {}

  /**
   * Destructor - Nothing to clean up
   *
   */
  virtual ~Torch_Interaction() = default;

  /**
   * Initializes everything necessary
   */
  virtual int init(topology::Topology &topo, configuration::Configuration &conf,
                   simulation::Simulation &sim, std::ostream &os = std::cout,
                   bool quiet = false) override;

  /**
   * Evaluates the Torch model and updates energies and forces
   */
  virtual int calculate_interactions(topology::Topology &topo,
                                     configuration::Configuration &conf,
                                     simulation::Simulation &sim) override;

protected:
  /**
   * Loads and deserializes the model
   */
  virtual int load_model();

  /**
   * Prepares the coordinates according to the atom selection scheme selected
   */
  virtual int prepare_input(const simulation::Simulation &sim) = 0;

  /**
   * Initializes tensors ready to go into the model
   */
  virtual int build_tensors(const simulation::Simulation &sim) = 0;

  /**
   * Forward pass of the model loaded
   */
  virtual int forward() = 0;

  /**
   * Backward pass of the model loaded
   */
  virtual int backward() = 0;

  /**
   * Gets energy from Torch
   */
  virtual int get_energy() const = 0;

  /**
   * Gets forces from Torch
   */
  virtual int get_forces() const = 0;

  /**
   * Saves the data to GROMOS
   */
  virtual int write_data(topology::Topology &topo,
                         configuration::Configuration &conf,
                         const simulation::Simulation &sim) const = 0;

  /**
   * Parameters of the model loaded
   */
  simulation::torch_model model;

  /**
   * A representation of the model
   */
  torch::jit::script::Module module;
};

} // namespace interaction

#endif /* INCLUDED_TORCH_INTERACTION_H */
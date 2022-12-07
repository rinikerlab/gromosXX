/**
 * @file torch_slim_interaction.h
 * torch_slim_interaction
 *
 */

#ifndef INCLUDED_TORCH_GLOBAL_INTERACTION_H
#define INCLUDED_TORCH_GLOBAL_INTERACTION_H

#include "../../stdheader.h"
#include "../../interaction/special/torch_interaction.h"
#include "../../simulation/parameter.h"
#include "../../simulation/simulation.h"
#include "../../configuration/configuration.h"

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/torch.h>

namespace interaction {

/**
 * A specialized version of Torch_Interaction that sends all atoms to a
 * Torch model, typename T indicates numerical precision (half, single, double)
 */
template <typename T>
class Torch_Global_Interaction : public Torch_Interaction<T> {

public:
  /**
   * Initializes the interaction
   */
  Torch_Global_Interaction(const simulation::torch_model &model)
      : Torch_Interaction<T>(model, "Torch Global Interface") {}

  /**
   * Deallocates resources
   */
  virtual ~Torch_Global_Interaction() = default;

  /**
   * Initializes everything necessary
   */
  virtual int init(topology::Topology &topo, configuration::Configuration &conf,
                   simulation::Simulation &sim, std::ostream &os = std::cout,
                   bool quiet = false) override;

private:

  /**
   * Prepares the coordinates according to the atom selection scheme selected
   */
  virtual int prepare_input(const topology::Topology& topo, 
                            const configuration::Configuration& conf, 
                            const simulation::Simulation& sim) override;

  /**
   * Initializes tensors ready to go into the model
   */
  virtual int build_tensors(const simulation::Simulation &sim) override;

  /**
   * Forward pass of the model loaded
   */
  virtual int forward() override;

  /**
   * Backward pass of the model loaded
   */
  virtual int backward() override;

  /**
   * Gets energy from Torch and updates Gromos
   */
  virtual int update_energy(topology::Topology &topo,
                            configuration::Configuration &conf,
                            const simulation::Simulation &sim) override;

  /**
   * Gets forces from Torch and updates Gromos
   */
  virtual int update_forces(topology::Topology &topo,
                            configuration::Configuration &conf,
                            const simulation::Simulation &sim) override;

  /**
   * Saves Torch input data
   */
  virtual void save_torch_input(const unsigned int step
                              , const topology::Topology& topo
                              , const configuration::Configuration& conf
                              , const simulation::Simulation& sim) override;

  /**
   * Saves Torch output data
   */
  virtual void save_torch_output(const unsigned int step
                               , const topology::Topology& topo
                               , const configuration::Configuration& conf
                               , const simulation::Simulation& sim) override;
  
  /**
   * Saves the coordinates sent to Torch
   */
  virtual void save_input_coord(std::ofstream& ifs
                               , const unsigned int step
                               , const topology::Topology& topo
                               , const configuration::Configuration& conf
                               , const simulation::Simulation& sim);

  /**
   * Saves the energy and gradients from backwards call on Torch model
   */
  virtual void save_output_gradients(std::ofstream& ifs
                                   , const unsigned int step
                                   , const topology::Topology& topo
                                   , const configuration::Configuration& conf
                                   , const simulation::Simulation& sim);

  /**
   * How many atoms are there in the system
  */
  unsigned natoms;

  /**
   * How many batches are sent to Torch each iteration
  */
  unsigned batch_size;

  /**
   * How many Cartesian coordinates
  */
  unsigned dimensions;

  /**
   * Handle to the input coordinate trajectory (Torch) 
   */
  mutable std::ofstream input_coordinate_stream;

  /**
   * Handle to the output gradient trajectory (Torch) 
   */
  mutable std::ofstream output_gradient_stream;

  /**
   * Atomic positions as C style array
   */
  std::vector<T> positions;

  /**
   * A (non-owning) tensor to hold atom positions
   */
  torch::Tensor positions_tensor;

  /**
   * A (non-owning) tensor to hold energies calculated
   */
  torch::Tensor energy_tensor;

  /**
   * A (non-owning) tensor to hold gradients calculated
   */
  torch::Tensor gradient_tensor;

};

} // namespace interaction

#endif /* INCLUDED_TORCH_GLOBAL_INTERACTION_H */
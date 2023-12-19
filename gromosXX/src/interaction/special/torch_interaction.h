/**
 * @file torch_interaction.h
 * torch
 */

#ifndef INCLUDED_TORCH_INTERACTION_H
#define INCLUDED_TORCH_INTERACTION_H

#include <vector>

#include <torch/torch.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include <torchcluster/cluster.h>

#include "../../stdheader.h"
#include "../../interaction/interaction.h"
#include "../../simulation/parameter.h"

namespace interaction {

/*
 * Provides an interface to PyTorch, typename T indicates numerical precision (half, single, double)
 */
template <typename T>
class Torch_Interaction : public Interaction {

public:
  /**
   * Constructor - Initializes name and timer of the Interaction
   *
   */
  Torch_Interaction(const simulation::torch_model &model,
                    const std::string &name)
      : Interaction(name), model(model) {
    // initializing where tensors will live and how they look like
    tensor_int64 = torch::TensorOptions() 
                  .dtype(torch::kInt64)
                  .layout(torch::kStrided); 
    tensor_float_gradient = torch::TensorOptions() 
                  .dtype(model.precision)
                  .layout(torch::kStrided)
                  .requires_grad(true);  
    tensor_float_no_gradient = torch::TensorOptions() 
                  .dtype(model.precision)
                  .layout(torch::kStrided)
                  .requires_grad(false);
  }

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
   * Initializes everything necessary
   */
  virtual int init_interaction(topology::Topology &topo, configuration::Configuration &conf,
                   simulation::Simulation &sim, std::ostream &os = std::cout,
                   bool quiet = false) = 0;
  /**
   * Loads and deserializes the model
   */
  virtual int load_model();

  /**
   * Prepares the coordinates according to the atom selection scheme selected
   */
  virtual int prepare_input(const topology::Topology& topo, 
                            const configuration::Configuration& conf, 
                            const simulation::Simulation& sim) = 0;

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
   * Gets energy from Torch and updates Gromos
   */
  virtual int update_energy(topology::Topology &topo,
                            configuration::Configuration &conf,
                            const simulation::Simulation &sim) = 0;

  /**
   * Gets forces from Torch and updates Gromos
   */
  virtual int update_forces(topology::Topology &topo,
                            configuration::Configuration &conf,
                            const simulation::Simulation &sim) = 0;

  /**
   * Saves Torch input data
   */
  virtual void save_torch_input(const topology::Topology& topo
                              , const configuration::Configuration& conf
                              , const simulation::Simulation& sim) = 0;

  /**
   * Saves Torch output data
   */
  virtual void save_torch_output(const topology::Topology& topo
                               , const configuration::Configuration& conf
                               , const simulation::Simulation& sim) = 0;
  
  /**
   * Helper function to write the current step size
   */
  virtual void write_step_size(std::ofstream& ifs, 
                               const unsigned int step) const;

  /**
   * Helper function to write the header in coordinate files
   */
  virtual void write_coordinate_header(std::ofstream& ifs) const;

  /**
   * Helper function to write the footer in coordinate files
   */
  virtual void write_coordinate_footer(std::ofstream& ifs) const;

  /**
   * Helper function to write a single gradient to a file
   */
  virtual void write_gradient(const math::Vec& gradient, 
                              std::ofstream& inputfile_stream) const;

  /**
   * Helper function to write a single QM atom to a file
   */
  virtual void write_atom(std::ofstream& inputfile_stream
                           , const int atomic_number
                           , const math::Vec& pos) const;

  /**
   * Helper function to open a file
   */
  virtual int open_input(std::ofstream& inputfile_stream, const std::string& input_file) const;

  /**
   * Print units conversion factors
   */
  virtual void print_unit_factors(std::ostream & os) const {
    os << model.unit_factor_length << ", "
       << model.unit_factor_energy << ", "
       << model.unit_factor_force << ", "
       << model.unit_factor_charge;
  };
  
  /**
   * Parameters of the model loaded
   */
  simulation::torch_model model;

  /**
   * A representation of the model
   */
  torch::jit::script::Module module;

  /**
   * Stores options how integer tensor are stored (e.g. for atomic indices) -> int64
  */
  torch::TensorOptions tensor_int64;

  /**
   * Stores options how floating point tensor are stored that require gradients (e.g. for atomic positions)
  */
  torch::TensorOptions tensor_float_gradient;

  /**
   * Stores options how floating point tensor are stored that require no gradients (e.g. for atomic charges)
  */
  torch::TensorOptions tensor_float_no_gradient;
#ifdef XXMPI
  /**
   * MPI rank of the current process
  */
  int m_rank;
  /**
   * Number of MPI ranks
  */
  int m_size;
#endif
};

} // namespace interaction

#endif /* INCLUDED_TORCH_INTERACTION_H */

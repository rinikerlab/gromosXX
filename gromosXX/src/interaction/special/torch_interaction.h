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
    Torch_Interaction(const simulation::torch_model& model) : Interaction("Torch"), model(model) {}

    /**
     * Destructor - Nothing to clean up
     * 
     */
    virtual ~Torch_Interaction() = default;

    /**
     * Loads up the model
     */
    int init(topology::Topology & topo,
		     configuration::Configuration & conf,
		     simulation::Simulation & sim,
		     std::ostream & os = std::cout,
		     bool quiet = false) override;


    protected:

      virtual int load_model();
      
      /**
       * Forward pass of the model loaded
       */
      virtual void forward() = 0;

      /**
       * Backward pass of the model loaded
       */
      virtual void backward() = 0;

      /**
       * Parameters on the model loaded
       */
      simulation::torch_model model;

      /**
       * A representation of the model
       */
      torch::jit::script::Module module;

      /**
       * energy calculated by PyTorch
       */
      double energy;

  };

} // interaction

#endif /* INCLUDED_TORCH_INTERACTION_H */
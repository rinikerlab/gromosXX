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


    protected:

      /**
       * Forward pass of the model loaded
       */
      virtual void forward() = 0;

      /**
       * Backward pass of the model loaded
       */
      virtual void backward() = 0;

      /**
       * Information on the model loaded
       */
      simulation::torch_model model;

      /**
       * energy calculated by PyTorch
       */
      double energy;

  };

} // interaction

#endif /* INCLUDED_TORCH_INTERACTION_H */
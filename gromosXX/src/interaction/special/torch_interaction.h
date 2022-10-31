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

namespace interaction {

  /*
   * Provides an interface to PyTorch
   */
  class Torch_Interaction : public Interaction {

  public:

    /**
     * Constructor
     * 
     */
    Torch_Interaction(const std::string& name) : Interaction(name) {}

    /**
     * Destructor
     * 
     */
    virtual ~Torch_Interaction() = default;

    private:

      torch::Tensor test_tensor;

      torch::Tensor test_tensor_2;

  };

} // interaction

#endif /* INCLUDED_TORCH_INTERACTION_H */
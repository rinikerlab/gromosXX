/**
 * @file torch_qmmm_interaction.cc
 * torch_qmmm_interaction
 * 
 */

#ifndef INCLUDED_TORCH_QMMM_INTERACTION_H
#define INCLUDED_TORCH_QMMM_INTERACTION_H

#include "../../stdheader.h"
#include "../../interaction/qmmm/qm_zone.h"
#include "../../interaction/special/torch_interaction.h"
#include "../../simulation/parameter.h"

#include <vector>

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

namespace interaction {

/**
 * A specialized version of Torch_Interaction that sends the QM/MM zone to a Torch model
*/
class Torch_QMMM_Interaction : public Torch_Interaction {

public:

    /**
     * Initializes the interaction
     */
    Torch_QMMM_Interaction(const simulation::torch_model& model);

    /**
     * Deallocates resources
    */
    virtual ~Torch_QMMM_Interaction();

    /**
     * Initializes everything necessary
     */
    virtual int init(topology::Topology & topo,
		     configuration::Configuration & conf,
		     simulation::Simulation & sim,
		     std::ostream & os = std::cout,
		     bool quiet = false) override;

private:

    /**
     * Prepares the coordinates according to the atom selection scheme selected
     */
    virtual int prepare_coordinates(simulation::Simulation& sim) override;

    /**
     * Initializes tensors ready to go into the model
     */
    virtual int prepare_tensors(simulation::Simulation& sim) override;

    /**
     * Sets-up a pointer to the QM zone of the simulation
    */
    int initialize_qm_zone();

    /**
     * Sets-up the vector with QM atom types
    */
    int initialize_qm_atom_types();

    /**
     * A (non-owning) pointer to the QM zone
    */
    QM_Zone* qm_zone_ptr;
    
    /**
     * Atomic numbers of the QM zone as C style array
    */
    std::vector<unsigned> qm_atomic_numbers;

    /**
     * Atomic positions of the QM zone as C style array
    */
    std::vector<float> qm_positions;

    /**
     * Atomic numbers of the MM zone as C style array
    */
    std::vector<unsigned> mm_atomic_numbers;

    /**
     * Charges of the MM zone as C style array
    */
    std::vector<float> mm_charges;

    /**
     * Atomic positions of the MM zone as C style array
    */
    std::vector<float> mm_positions;

    /**
     * A (non-owning) tensor to hold QM atomic numbers
    */
    torch::Tensor qm_atomic_numbers_tensor;

    /**
     * A (non-owning) tensor to hold QM positions
    */
    torch::Tensor qm_positions_tensor;

    /**
     * A (non-owning) tensor to hold MM atomic positions
    */
    torch::Tensor mm_atomic_numbers_tensor;

    /**
     * A (non-owning) tensor to hold MM charges
    */
    torch::Tensor mm_charges_tensor;

    /**
     * A (non-owning) tensor to hold MM positions
    */
    torch::Tensor mm_positions_tensor;

};

} // namespace interaction

#endif /* INCLUDED_TORCH_QMMM_INTERACTION_H */
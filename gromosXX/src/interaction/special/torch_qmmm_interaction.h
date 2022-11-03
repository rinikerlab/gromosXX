/**
 * @file torch_qmmm_interaction.cc
 * torch_qmmm_interaction
 * 
 */

#ifndef INCLUDED_TORCH_QMMM_INTERACTION_H
#define INCLUDED_TORCH_QMMM_INTERACTION_H

#include "../../stdheader.h"
#include "../../interaction/qmmm/qm_zone.h"
#include "../../interaction/qmmm/qmmm_interaction.h"
#include "../../interaction/special/torch_interaction.h"
#include "../../simulation/parameter.h"
#include "../../simulation/simulation.h"
#include "../../configuration/configuration.h"

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
    Torch_QMMM_Interaction(const simulation::torch_model& model) : Torch_Interaction(model, "Torch QM/MM Interface") {}

    /**
     * Deallocates resources
    */
    virtual ~Torch_QMMM_Interaction() = default;

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
     * Sets-up a pointer to the QM zone of the simulation
    */
    virtual int init_qm_zone();

    /**
     * Prepares the coordinates according to the atom selection scheme selected
     */
    virtual int prepare_input(const simulation::Simulation& sim) override;

    /**
     * Sets-up the vector with QM atom types
    */
    virtual int init_qm_atom_numbers();

    /**
     * Sets-up the vector with QM coordinates - will also cast double to float
    */
    virtual int prepare_qm_atoms();

    /**
     * Sets-up the vectors with MM atom types, charges, and coordinates - will also cast double to float
    */
    virtual int prepare_mm_atoms();

    /**
     * Initializes tensors ready to go into the model
     */
    virtual int build_tensors(const simulation::Simulation& sim) override;

    /**
      * Forward pass of the model loaded
      */
    virtual int forward() override;

    /**
     * Backward pass of the model loaded
     */
    virtual int backward() override;

    /**
     * Gets energy from Torch
    */
    virtual int get_energy() override;

    /**
     * Gets forces from Torch
    */
    virtual int get_forces() override;

    /**
      * Saves the data to GROMOS
     */
    virtual int write_data(topology::Topology & topo,
				                   configuration::Configuration & conf,
				                   const simulation::Simulation & sim) override;

    /**
     * Computes the number of charges like in QM_Worker.cc
    */
    virtual int get_num_charges(const simulation::Simulation& sim);

    /**
     * A (non-owning) pointer to the QM zone
    */
    QM_Zone* qm_zone_ptr;

    /**
     * A (non_owning) pointer to the QMMM interaction
    */
    QMMM_Interaction* qmmm_ptr;

    /**
     * The size of the QM zone (QM atoms + QM link atoms)
    */
    int natoms;

    /**
     * Number of point charges in the MM zone
    */
    int ncharges;
    
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

    /**
     * A (non-owning) tensor to hold energies calculated
    */
    torch::Tensor energy_tensor;

    /**
     * A (non-owning) tensor to hold QM gradients calculated
    */
    torch::Tensor qm_gradient_tensor;

    /**
     * A (non-owning) tensor to hold MM gradients calculated
    */
    torch::Tensor mm_gradient_tensor;

};

} // namespace interaction

#endif /* INCLUDED_TORCH_QMMM_INTERACTION_H */
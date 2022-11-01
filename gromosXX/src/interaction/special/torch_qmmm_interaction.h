/**
 * @file torch_qmmm_interaction.cc
 * torch_qmmm_interaction
 * 
 */

#ifndef INCLUDED_TORCH_QMMM_INTERACTION_H
#define INCLUDED_TORCH_QMMM_INTERACTION_H

#include "../../stdheader.h"
#include "../../interaction/special/torch_interaction.h"
#include "../../simulation/parameter.h"

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

    int init(topology::Topology & topo,
		     configuration::Configuration & conf,
		     simulation::Simulation & sim,
		     std::ostream & os = std::cout,
		     bool quiet = false) override;

    int calculate_interactions(topology::Topology & topo,
				               configuration::Configuration & conf,
				               simulation::Simulation & sim) override;

private:

    virtual void forward() override;

    virtual void backward() override;

};

} // namespace interaction

#endif /* INCLUDED_TORCH_QMMM_INTERACTION_H */
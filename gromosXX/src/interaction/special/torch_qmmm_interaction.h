/**
 * @file torch_qmmm_interaction.cc
 * torch_qmmm_interaction
 * 
 */

#ifndef INCLUDED_TORCH_QMMM_INTERACTION_H
#define INCLUDED_TORCH_QMMM_INTERACTION_H

#include "../../stdheader.h"
#include "../../interaction/special/torch_interaction.h"

namespace interaction {

/**
 * A specialized version of Torch_Interaction that sends the QM/MM zone to a Torch model
*/
class Torch_QMMM_Interaction : public Torch_Interaction {

public:

    /**
     * Initializes the interaction
    */
    Torch_QMMM_Interaction(const std::string& name);

    /**
     * Deallocates resources
    */
    virtual ~Torch_QMMM_Interaction();

private:



};

} // namespace interaction

#endif /* INCLUDED_TORCH_QMMM_INTERACTION_H */
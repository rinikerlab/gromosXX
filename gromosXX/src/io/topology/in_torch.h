/**
 * @file in_torch.h
 * read in a Torch specification file
 */
/**
 * @page torch Torch specification format
 * @date 01-11-2022
 *
 * A Torch specification file may contain the following
 * blocks:
 * - @ref title
 * - @ref blocks
 */

#ifndef INCLUDED_IN_TORCH_H
#define INCLUDED_IN_TORCH_H

#include "../instream.h"

namespace io {

  /**
   * @class In_Torch
   * reads in a Torch specification file
   */
  class In_Torch : public GInStream {

  public:
    /**
     * Default constructor.
     */
    In_Torch() {}
    /**
     * Constructor.
     */
    In_Torch(std::istream& is) : GInStream(is) { readStream(); };
    /**
     * Read in a Torch specification file.
     */
    void read(topology::Topology &topo,
	      simulation::Simulation & sim,
	      std::ostream & os = std::cout);
    /**
     * Reads in the models
     */
    void read_models(simulation::Simulation & sim);
  };
} // io

#endif // INCLUDED_IN_TORCH_H
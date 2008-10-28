/**
 * @file in_posres.h
 * read in a position restraining file.
 */
/**
 * @page posres position restraints format
 * @date 28-10-2008
 *
 * A position restraints/constraints specifcation file may contain the following
 * blocks:
 * - @ref title
 * - @ref posresspec
 * - @ref refposition
 * - @ref bfactor
 */

#ifndef INCLUDED_IN_POSRES_H
#define INCLUDED_IN_POSRES_H

#include "../instream.h"

namespace io {

  /**
   * @class In_Posres
   * reads in a position restraining file
   */
  class In_Posres : public GInStream {

  public:
    /**
     * Default constructor.
     */
    In_Posres() {}
    /**
     * Constructor.
     */
    In_Posres(std::istream& is) : GInStream(is) { readStream(); };
    /**
     * Read in a position restraining file.
     */
    void read(topology::Topology &topo,
	      simulation::Simulation & sim,
	      std::ostream & os = std::cout);

  };

} // io

#endif

/**
 * @file InPerturbationTopology.h
 * read in a perturbation topology file (03 format)
 */

#ifndef INCLUDED_INPERTURBATIONTOPOLOGY_H
#define INCLUDED_INPERTURBATIONTOPOLOGY_H

namespace io
{
  /**
   * @class InPerturbationTopology
   * reads in a perturbation topology file (03 version)
   * and parses it into simulation::Perturbation_Topology
   * @sa simulation::Perturbation_Topology
   */
  class InPerturbationTopology : public GInStream 
  {
  public:
    /**
     * Constructor.
     */
    InPerturbationTopology(std::istream &is);
    /**
     * parse the topology.
     */
    InPerturbationTopology & 
    operator>>(simulation::Perturbation_Topology &topo);
    
  private:
    /**
     * read the stream.
     */
    void read_stream();
    
    /**
     * store the blocks.
     */
    std::map<std::string, std::vector<std::string> > m_block;
  };
  
} // io

// inline methods
#include "InPerturbationTopology.tcc"

#endif

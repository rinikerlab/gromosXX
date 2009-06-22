/**
 * @file xray_restraint_interaction.h
 * xray restraining
 */

#ifndef INCLUDED_XRAY_RESTRAINT_INTERACTION_H
#define INCLUDED_XRAY_RESTRAINT_INTERACTION_H

// Additional Clipper Headers
#ifdef HAVE_CLIPPER
#include <clipper/clipper.h>
#include <clipper/clipper-ccp4.h>
#include <clipper/clipper-contrib.h>
#endif

namespace interaction {

  /**
   * @class xray_restraint_interaction
   * calculates the xray restraining interaction
   */
  class Xray_Restraint_Interaction : public Interaction {
  public:

    /**
     * Constructor.
     */
    Xray_Restraint_Interaction();
    /**
     * Destructor.
     */
    virtual ~Xray_Restraint_Interaction();

    /**
     * init
     */
    virtual int init(topology::Topology &topo,
            configuration::Configuration &conf,
            simulation::Simulation &sim,
            std::ostream &os = std::cout,
            bool quiet = false);
    /**
     * calculate the interactions.
     */
    virtual int calculate_interactions(topology::Topology & topo,
            configuration::Configuration & conf,
            simulation::Simulation & sim);

  protected:
#ifdef HAVE_CLIPPER
    /**
     * pointer to the atoms
     */
    clipper::Atom_list atoms;
    /**
     * the HKLs
     */
    clipper::HKL_info hkls;
    /**
     * the structure factors
     */
    clipper::HKL_data<clipper::data32::F_phi> fphi;
    /**
     * copy of fphi for print of electron-density maps
     */
    clipper::HKL_data<clipper::data32::F_phi> fphi_print;
    /**
     * the gradients
     */
    clipper::HKL_data<clipper::data32::F_phi> D_k;
    /**
     * the map for the gradient convolution
     */
    clipper::Xmap<clipper::ftype32> d_r;
#endif

    template<math::boundary_enum B, math::virial_enum V>
    void _calculate_xray_restraint_interactions
    (topology::Topology & topo,
            configuration::Configuration & conf,
            simulation::Simulation & sim,
            int & error);
  };

} // interaction
#endif


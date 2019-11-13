/**
 * @file qm_zone.h
 * QM zone class - implements QM zone for QM/MM calculations
 */

#ifndef INCLUDED_QM_ZONE_H
#define	INCLUDED_QM_ZONE_H

namespace math
{
  template<math::boundary_enum B>
  class Periodicity;
}

namespace interaction {
  class QM_Atom;
  class MM_Atom;
  class QM_Link;
  class PairlistContainer;
  /**
   * @class QM_Zone
   * Holds all information on QM Zone - this will be passed between QM worker and QMMM interaction
   * merges function of QM storage and QM zone and eventually allows multiple QM zones
   */
  class QM_Zone {
  public:
    /**
     * Constructor
     */
    explicit QM_Zone(int net_charge = 0
                   , int spin_multiplicity = 1
                  );

    /**
     * Copy constructor
     */
    /*explicit QM_Zone(const QM_Zone& qmz
                   , int net_charge
                   , int spin_multiplicity
                  );*/

    /**
     * Assignment operator
     */
    //QM_Zone& operator= (const QM_Zone& qmz);

    /**
     * Destructor
     */
    ~QM_Zone();

    /**
     * the QM energy of the zone
     */
    double QM_energy;

    /**
     * the MM energy of the zone - convenient for subtractive scheme
     */
    double MM_energy;

    /**
     * QM atoms
     */
    std::set<interaction::QM_Atom> qm;

    /**
     * MM atoms and point charges (including COS) - these are expected to be rebuilt between steps (due to cutoff)
     */
    std::set<interaction::MM_Atom> mm;

    /**
     * QMMM links
     */
    std::set<interaction::QM_Link> link;
    
    /**
     * Initialize QM zone
     */
    int init(const topology::Topology& topo, 
             const configuration::Configuration& conf, 
             const simulation::Simulation& sim);
    
    /**
     * Update positions of QM and re-gather MM atoms
     */
    int update(const topology::Topology& topo, 
               const configuration::Configuration& conf, 
               const simulation::Simulation& sim);

    /**
     * Write positions to the configuration
     */
    void write_pos(math::VArray& pos);

    /**
     * Write forces to the configuration
     */
    void write_force(math::VArray& force);

    /**
     * Write charges to the topology
     */
    void write_charge(math::SArray& charge);
    
    /**
     * Update QM-MM pairlist
     */
    void update_pairlist(const topology::Topology& topo
                       , const simulation::Simulation& sim
                       , PairlistContainer& pairlist
                       , unsigned begin, unsigned end
                       , unsigned stride) const;
  protected:
    /**
     * Zero energies
     */
    void zero();

    /**
     * Clear the QM zone
     */
    void clear();

    /**
     * Scale charges with distance to nearest QM atom
     */
    void scale_charges(const simulation::Simulation& sim);

    /**
     * Gather QM atoms
     */
    int get_qm_atoms(const topology::Topology& topo, 
                     const configuration::Configuration& conf, 
                     const simulation::Simulation& sim);

    /**
     * Update positions of QM atoms - wrapper
     */
    int update_qm_pos(const topology::Topology& topo, 
                      const configuration::Configuration& conf, 
                      const simulation::Simulation& sim);

    /**
     * Gather MM atoms - wrapper
     */
    int get_mm_atoms(const topology::Topology& topo, 
                     const configuration::Configuration& conf, 
                     const simulation::Simulation& sim);
    
    /**
     * Get QM-MM bonds
     */
    void get_links(const topology::Topology& topo, 
                   const simulation::Simulation& sim);
    
    /**
     * Update caps positions
     */
    void update_links(const simulation::Simulation& sim);

    /**
     * Update positions of QM atoms - internal function
     */
    template<math::boundary_enum B>
    int _update_qm_pos(const topology::Topology& topo, 
                       const configuration::Configuration& conf, 
                       const simulation::Simulation& sim);

    /**
     * Gather MM atoms (chargegroup-based cutoff) - internal function
     */
    template<math::boundary_enum B>
    int _get_mm_atoms(const topology::Topology& topo, 
                      const configuration::Configuration& conf, 
                      const simulation::Simulation& sim);

    /**
     * Gather MM atoms (atom-based cutoff) - internal function
     */
    template<math::boundary_enum B>
    int _get_mm_atoms_atomic(const topology::Topology& topo, 
                             const configuration::Configuration& conf, 
                             const simulation::Simulation& sim);

    /**
     * Gather linked MM atoms - internal function
     */
    template<math::boundary_enum B>
    void get_linked_mm_atoms(const topology::Topology& topo, 
                            const configuration::Configuration& conf,
                            const math::Periodicity<B>& periodicity);
  };
}
#endif	/* QM_ZONE_H */


/**
 * @file gaussian_worker.h
 * The worker class for the Gaussian QM software
 */
#ifndef INCLUDED_GAUSSIAN_WORKER_H
#define	INCLUDED_GAUSSIAN_WORKER_H

#include <simulation/simulation.h>

#include <interaction/qmmm/qm_worker.h>
#include <interaction/qmmm/qm_zone.h>

namespace interaction {
  /**
   * @class Gaussian_Worker
   * a worker class which calls the Gaussian software
   */
  class Gaussian_Worker : public QM_Worker {
  public:
    /**
     * Constructor
     */
    Gaussian_Worker();
    /**
     * Destructor
     */
    virtual ~Gaussian_Worker() = default;
    /**
     * initialise the QM worker
     * @return 0 if successful, non-zero on failure
     */
    virtual int init(simulation::Simulation& sim);

  private:
    /**
     * Pointer to simulation parameters
     */
    simulation::Parameter::qmmm_struct::gaussian_param_struct* param;

    /**
     * Write input file for QM
     * @param topo Topology
     * @param conf Configuration
     * @param sim Simulation
     * @param qm_zone QM Zone
     */
    int write_input(const topology::Topology& topo
                  , const configuration::Configuration& conf
                  , const simulation::Simulation& sim
                  , const interaction::QM_Zone& qm_zone);

    /**
     * System call
     */
    int system_call();

    /**
     * Read outputs
     */
    int read_output(topology::Topology& topo
                  , configuration::Configuration& conf
                  , simulation::Simulation& sim
                  , interaction::QM_Zone& qm_zone);

    /**
     * Write QM atom
     */
    void write_qm_atom(std::ofstream& inputfile_stream
                  , const int atomic_number
                  , const math::Vec& pos);

    /**
     * Write MM atom
     */
    void write_mm_atom(std::ofstream& inputfile_stream
                      , const math::Vec& pos
                      , const double charge);

    /**
     * Write MM position
     */
    void write_mm_pos(std::ofstream& inputfile_stream
                      , const math::Vec& pos);

    /**
     * Parse charges
     */
    int parse_charges(std::ifstream& ofs, interaction::QM_Zone& qm_zone);

    /**
     * Parse coordinates
     */
    int parse_coordinates(std::ifstream& ofs, interaction::QM_Zone& qm_zone);

    /**
     * Parse energy
     */
    int parse_energy(std::ifstream& ofs, interaction::QM_Zone& qm_zone);

    /**
     * Parse gradients wrapper
     */
    int parse_gradients(const simulation::Simulation& sim
                      , std::ifstream& ofs
                      , interaction::QM_Zone& qm_zone);

    /**
     * Parse gradients
     */
    template<class AtomType>
    int _parse_gradients(std::ifstream& ofs, std::set<AtomType>& atom_set);

    /**
     * Parse gradient line
     */
    int parse_gradient(std::ifstream& ofs
                     , math::Vec& force
                     , const double unit_factor);
  };

  /**
   * Parse gradients of MM atoms
   */
  template<>
  int Gaussian_Worker::_parse_gradients<interaction::MM_Atom>
        (std::ifstream& ofs, std::set<interaction::MM_Atom>& atom_set);
}

#endif	/* GAUSSIAN_WORKER_H */


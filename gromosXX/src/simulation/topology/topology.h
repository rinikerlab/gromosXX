/**
 * @file topology.h
 * the topology class
 */

#ifndef INCLUDED_TOPOLOGY_H
#define INCLUDED_TOPOLOGY_H

namespace simulation
{
  class Multibath;
}

namespace simulation
{
  /**
   * @class Topology
   * holds the topological information of
   * the simulated system
   * @sa simulation::simulation
   * @sa simulation::system
   */
  class Topology
  {
  public:
    /**
     * Constructor
     */
    explicit Topology();

    /**
     * integer atom code accessor.
     */
    int iac(size_t const i)const;
    
    /**
     * masses accessor
     */
    math::SArray &mass();

    /**
     * masses const accessor
     */
    math::SArray const & mass()const;

    /**
     * charge accessor
     */
    math::SArray &charge();
    
    /**
     * charge const accessor
     */
    math::SArray const & charge()const;
    
    /**
     * soluteatom accessor.
     */
    Solute & solute();

    /**
     * const solute accessor.
     */
    Solute const & solute()const;
    
    /**
     * number of solvents.
     */
    size_t num_solvents()const;
    
    /**
     * solvent accessor.
     * support for multiple solvents.
     */
    Solvent & solvent(size_t i);
    /**
     * const solvent accessor.
     * support for multiple solvents.
     */
    Solvent const & solvent(size_t i)const;

    /**
     * add a solvent.
     */
    void add_solvent(Solvent solv);

    /**
     * add solvent to the simulation.
     * @param solv the solvent (multiple solvents).
     * @param num_molecules the number of solvent molecules to add.
     */
    void solvate(size_t solv, size_t num_molecules);
    
    /**
     * set the capacity of solute atoms
     */
    void resize(size_t const atoms);

    /**
     * get the total number of atoms.
     */
    size_t num_atoms()const;

    /**
     * get the number of solute atoms
     */
    size_t num_solute_atoms()const;

    /**
     * get the total number of solvent atoms.
     */
    size_t num_solvent_atoms()const;

    /**
     * get the number of solvent molecules.
     */
    size_t num_solvent_molecules(size_t i)const;
    
    /**
     * get the number of solvent atoms.
     */
    size_t num_solvent_atoms(size_t i)const;

    /**
     * add a solute atom.
     */
    void add_solute_atom(std::string name, int residue_nr, int iac,
			 double mass, double charge, bool chargegroup,
			 std::set<int> exclusions,
			 std::set<int> one_four_pairs);
    
    /**
     * residue names.
     */
    std::vector<std::string> & residue_name();

    /**
     * all exclusions for atom i. Exclusions and 1,4 interactions.
     */
    std::set<int> & all_exclusion(size_t const i);

    /**
     * const all exclusions for atom i. Exclusions and 1,4 interactions.
     */
    std::set<int> const & all_exclusion(size_t const i)const;
    /**
     * exclusions for atom i.
     */
    std::set<int> & exclusion(size_t const i);
    /**
     * exclusions
     */
    std::vector<std::set<int> > & exclusion();
    
    /**
     * 1,4 pairs of atom i.
     */
    std::set<int> & one_four_pair(size_t const i);
    /**
     * 1,4 pairs 
     */
    std::vector<std::set<int> > & one_four_pair();
    
    /**
     * the number of chargegroups present.
     */
    size_t num_chargegroups()const;
    /**
     * the number of solute chargegroups.
     */
    size_t num_solute_chargegroups()const;
    /**
     * iterator over the chargegrops
     */
    chargegroup_iterator chargegroup_begin()const;
    /**
     * end of the chargegroup iterator.
     */
    chargegroup_iterator chargegroup_end()const;
    /**
     * the molecules.
     */
    std::vector<size_t> & molecules();
    /**
     * iterator over the molecules.
     */
    Molecule_Iterator molecule_begin();
    /**
     * end of molecule iterator.
     */
    Molecule_Iterator molecule_end();

    /**
     * const energy group accessor.
     */
    std::vector<size_t> const & energy_groups()const;

    /**
     * energy group accessor.
     */
    std::vector<size_t> & energy_groups();

    /**
     * const energy group of atoms accessor.
     */
    std::vector<size_t> const & atom_energy_group()const;

    /**
     * energy group of atoms accessor.
     */
    std::vector<size_t> & atom_energy_group();
  
    /**
     * energy group of atom accessor
     */
    const size_t atom_energy_group(size_t i)const;

    /**
     * calculate constraint degrees of freedom
     */
    void calculate_constraint_dof(simulation::Multibath &multibath)const;
    
    /**
     * check state
     */
    int check_state()const;

  private:
    /**
     * the soluteatoms.
     */
    Solute m_solute;

    /**
     * the number of solvent molecules.
     */
    std::vector<size_t> m_num_solvent_molecules;
    
    /**
     * the number of solvent atoms.
     * vector for multiple solvents.
     */
    std::vector<size_t> m_num_solvent_atoms;
    
    /**
     * the solvents (multiple solvent).
     */
    std::vector<Solvent> m_solvent;
    
    /**
     * the integer atom code.
     */
    std::vector<int> m_iac;

    /**
     * the atom masses.
     */
    math::SArray m_mass;
    
    /**
     * the atom charges.
     */
     math::SArray m_charge;

    /**
     * the atom exclusions.
     */
    std::vector< std::set<int> > m_exclusion;
    
    /**
     * the atom 1-4 interactions.
     */
    std::vector< std::set<int> > m_one_four_pair;
    
    /**
     * atom exclusions and 1-4 interactions.
     */
    std::vector< std::set<int> > m_all_exclusion;
    
    /**
     * the molecules.
     */
    std::vector<size_t> m_molecule;

    /**
     * the chargegroups.
     */
    std::vector<int> m_chargegroup;
        
    /**
     * the number of solute chargegroups.
     */
    size_t m_num_solute_chargegroups;
    
    /**
     * residue names (solute and solvent).
     */
    std::vector<std::string> m_residue_name;

    /**
     * energy groups.
     */
    std::vector<size_t> m_energy_group;
    
    /**
     * energy group of atom
     */
    std::vector<size_t> m_atom_energy_group;

    
  }; // topology
  
} // simulation

// inline method definitions
// #include "topology.tcc"

#endif

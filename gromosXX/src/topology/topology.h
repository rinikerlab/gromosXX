/**
 * @file topology.h
 * the topology class
 */

#ifndef INCLUDED_TOPOLOGY_H
#define INCLUDED_TOPOLOGY_H

#include <gromosXX/topology/core/core.h>
#include <gromosXX/topology/solute.h>
#include <gromosXX/topology/solvent.h>
#include <gromosXX/topology/perturbed_atom.h>
#include <gromosXX/topology/perturbed_solute.h>

namespace simulation
{
  class Multibath;
  class Simulation;
  class Parameter;
}

namespace topology
{
  /**
   * @class Topology
   * holds the topological information of
   * the simulated system
   */
  class Topology
  {
  public:
    /**
     * Constructor
     */
    explicit Topology();

    /**
     * copy constructor
     * can multiply topologies
     */
    explicit Topology(Topology const & topo, int mul_solute = 1, int mul_solvent = -1);
    
    /**
     * Destructor
     */
    ~Topology();

    /**
     * integer atom code accessor.
     */
    int iac(unsigned int const i)const {assert(i < m_iac.size()); return m_iac[i];}
    
    /**
     * masses accessor
     */
    math::SArray &mass() {return m_mass;}

    /**
     * masses const accessor
     */
    math::SArray const & mass()const {return m_mass;}

    /**
     * mass of atom i
     */
    double mass(int i)const 
    {
      return m_mass(i);
    }
    
    /**
     * charge accessor
     */
    math::SArray &charge() {return m_charge;}
    
    /**
     * charge const accessor
     */
    math::SArray const & charge()const{return m_charge;}

    double charge(int i)const
    {
      return m_charge(i);
    }
    
    /**
     * solute accessor.
     */
    Solute & solute() {return m_solute;}

    /**
     * perturbed solute accessor.
     */
    Perturbed_Solute & perturbed_solute() {return m_perturbed_solute;}

    /**
     * const solute accessor.
     */
    Solute const & solute()const{return m_solute;}

    /**
     * const perturbed solute accessor.
     */
    Perturbed_Solute const & perturbed_solute()const{return m_perturbed_solute;}
    
    /**
     * number of solvents.
     */
    unsigned int num_solvents()const {return unsigned(m_num_solvent_molecules.size());}
    
    /**
     * solvent accessor.
     * support for multiple solvents.
     */
    Solvent & solvent(unsigned int i) {assert(i < m_solvent.size()); return m_solvent[i];}
    /**
     * const solvent accessor.
     * support for multiple solvents.
     */
    Solvent const & solvent(unsigned int i)const
    {assert(i < m_solvent.size()); return m_solvent[i];}

    /**
     * add a solvent.
     */
    void add_solvent(Solvent solv) {m_solvent.push_back(solv);}

    /**
     * add solvent to the simulation.
     * @param solv the solvent (multiple solvents).
     * @param num_molecules the number of solvent molecules to add.
     */
    void solvate(unsigned int solv, unsigned int num_molecules);
    
    /**
     * set the capacity of solute atoms
     */
    void resize(unsigned int const atoms);

    /**
     * get the total number of atoms.
     */
    unsigned int num_atoms()const {return num_solute_atoms() + num_solvent_atoms();}

    /**
     * get the number of solute atoms
     */
    unsigned int num_solute_atoms()const{return solute().num_atoms();}

    /**
     * get the total number of solvent atoms.
     */
    unsigned int num_solvent_atoms()const;

    /**
     * get the number of solvent molecules.
     */
    unsigned int num_solvent_molecules(unsigned int i)const {
      assert(i < m_num_solvent_molecules.size());
      return m_num_solvent_molecules[i];
    }
    
    /**
     * get the number of solvent atoms.
     */
    unsigned int num_solvent_atoms(unsigned int i)const{
      assert(i<m_num_solvent_atoms.size());
      return m_num_solvent_atoms[i];
    }

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
    std::vector<std::string> & residue_names() {return m_residue_name;}
    /**
     * const residue names.
     */
    std::vector<std::string> const & residue_names()const {return m_residue_name;}

    /**
     * all exclusions for atom i. Exclusions and 1,4 interactions.
     */
    std::set<int> & all_exclusion(unsigned int const i){
      assert(i < m_all_exclusion.size());
      return m_all_exclusion[i];
    }

    /**
     * const all exclusions for atom i. Exclusions and 1,4 interactions.
     */
    std::set<int> const & all_exclusion(unsigned int const i)const{
      assert(i < m_all_exclusion.size());
      return m_all_exclusion[i];
    }

    /**
     * exclusions for atom i.
     */
    std::set<int> & exclusion(unsigned int const i){
      assert(i < m_exclusion.size());
      return m_exclusion[i];
    }
    /**
     * exclusions
     */
    std::vector<std::set<int> > & exclusion() {return m_exclusion;}
    
    /**
     * 1,4 pairs of atom i.
     */
    std::set<int> & one_four_pair(unsigned int const i){
      assert(i < m_one_four_pair.size());
      return m_one_four_pair[i];
    }
    /**
     * 1,4 pairs 
     */
    std::vector<std::set<int> > & one_four_pair(){return m_one_four_pair;}
    
    /**
     * chargegroup accessor.
     */
    std::vector<int> const & chargegroups()const { return m_chargegroup; }
    
    int chargegroup(int i)const 
    {
      assert(m_chargegroup.size() > i);
      return m_chargegroup[i];
    }

    /**
     * the number of chargegroups present.
     */
    unsigned int num_chargegroups()const {return unsigned(m_chargegroup.size())-1;}

    /**
     * the number of solute chargegroups.
     */
    unsigned int num_solute_chargegroups()const {return m_num_solute_chargegroups;}
    /**
     * iterator over the chargegrops
     */
    Chargegroup_Iterator chargegroup_begin()const {
      return Chargegroup_Iterator(m_chargegroup.begin());}

    /**
     * iterator on a specified chargegroup
     */
    Chargegroup_Iterator chargegroup_it(const unsigned int i)const
    {
      return Chargegroup_Iterator(m_chargegroup.begin()+i);
    }

    /**
     * end of the chargegroup iterator.
     */
    Chargegroup_Iterator chargegroup_end()const{
      return Chargegroup_Iterator(m_chargegroup.end()-1);}

    /**
     * the molecules.
     */
    std::vector<unsigned int> & molecules(){ return m_molecule;}
    /**
     * the molecules (const)
     */
    std::vector<unsigned int> const & molecules()const{ return m_molecule;}
    /**
     * iterator over the molecules.
     */
    Molecule_Iterator molecule_begin()const{
      return Molecule_Iterator(m_molecule.begin());};
    /**
     * end of molecule iterator.
     */
    Molecule_Iterator molecule_end()const{
      return Molecule_Iterator(m_molecule.end()-1);}

    /**
     * const energy group accessor.
     */
    std::vector<unsigned int> const & energy_groups()const {return m_energy_group;}

    /**
     * energy group accessor.
     */
    std::vector<unsigned int> & energy_groups() {return m_energy_group;}

    /**
     * const energy group of atoms accessor.
     */
    std::vector<unsigned int> const & atom_energy_group()const{
      return m_atom_energy_group;
    }

    /**
     * energy group of atoms accessor.
     */
    std::vector<unsigned int> & atom_energy_group(){
      return m_atom_energy_group;
    }
  
    /**
     * energy group of atom accessor
     */
    unsigned int atom_energy_group(unsigned int i)const{
      assert(i < m_atom_energy_group.size());
      return m_atom_energy_group[i];
    };

    /**
     * calculate constraint degrees of freedom
     */
    void calculate_constraint_dof(simulation::Multibath & mb,
				  bool rottrans_constraints)const;
    
    /**
     * check state
     */
    int check_state()const;

    /**
     * return lambda
     */
    double lambda()const {return m_lambda;}

    /**
     * set lambda
     */
    void lambda(double const l){ m_old_lambda = m_lambda; m_lambda = l;}

    /**
     * old lambda
     */
    double old_lambda()const { return m_old_lambda; }
    
    /**
     * return nlam (exponent to lambda)
     */
    int lambda_exp()const {return m_lambda_exp;}

    /**
     * set lambda exponent.
     */
    void lambda_exp(int const l) { m_lambda_exp = l; }

    /**
     * is the atom perturbed?
     */
    bool is_perturbed(unsigned int const i)const {
      if (i >= num_solute_atoms()) return false;
      
      assert(i < m_is_perturbed.size()); 
      return m_is_perturbed[i];
    }
    /**
     * is the atom perturbed?
     */
    std::vector<bool> & is_perturbed() { return m_is_perturbed;}
    /**
     * recalculate lambda dependent properties.
     */
    void update_for_lambda();

    /**
     * interaction matrix for scaled interactions with energy groups
     */
    std::map<std::pair<int, int>, std::pair<double, double> > const &
    energy_group_scaling()const { return m_energy_group_scaling;}

    /**
     * interaction matrix for scaled interactions with energy groups
     */
    std::map<std::pair<int, int>, std::pair<double, double> > &
    energy_group_scaling() { return m_energy_group_scaling;}

    /**
     * matrix for changed lambda dependence for interactions
     * between energy groups  energy groups
     */
    std::map<std::pair<int, int>, std::pair<int, double> > const &
    energy_group_lambdadep()const { return m_energy_group_lambdadep;}

    /**
     * matrix for changed lambda dependence for interactions
     * between energy groups  energy groups
     */
    std::map<std::pair<int, int>, std::pair<int, double> > &
    energy_group_lambdadep() { return m_energy_group_lambdadep;}

    /**
     * lambda prime's for different lambda dependencies.
     * (so that they do not have to be recalculated for every interaction.
     */
    std::vector<double> & lambda_prime() { return m_lambda_prime; }
    
    /**
     * d lambda prime / d lambda derivative.
     * (to store the d Energy / d lambda derivative in one energy class as
     * sum d E / d lambda prime * d lambda prime / d lambda 
     * for all lambda prime).
     */
    std::vector<double> & lambda_prime_derivative() 
    { return m_lambda_prime_derivative; }
    
    /**
     * alpha parameters for the perturbed energy derivatives.
     */
    std::vector<double> const & perturbed_energy_derivative_alpha()const
    { return m_perturbed_energy_derivative_alpha; }

    /**
     * alpha parameters for the perturbed energy derivatives.
     */
    std::vector<double> & perturbed_energy_derivative_alpha()
    { return m_perturbed_energy_derivative_alpha; }
    
    /**
     * position restraints accessor.
     */
    std::vector<position_restraint_struct> & position_restraints()
    {
      return m_position_restraint;
    }

    /**
     * const position restraints accessor.
     */
    std::vector<position_restraint_struct> const & position_restraints()const
    {
      return m_position_restraint;
    }

    /**
     * j value restraints accessor.
     */
    std::vector<jvalue_restraint_struct> & jvalue_restraints()
    {
      return m_jvalue_restraint;
    }

    /**
     * const jvalue restraints accessor.
     */
    std::vector<jvalue_restraint_struct> const & jvalue_restraints()const
    {
      return m_jvalue_restraint;
    }

    /**
     * multi cell topology accessor
     */
    Topology & multicell_topo()
    {
      assert(m_multicell_topo != NULL);
      return *m_multicell_topo;
    }

    /**
     * multi cell topology const accessor
     */
    Topology const & multicell_topo()const
    {
      return *m_multicell_topo;
    }

    /**
     * initialise the topology.
     * - adjust submolecules if empty
     */
    void init(simulation::Simulation const & sim, bool quiet = false);
    
  private:
    /**
     * the solute.
     */
    Solute m_solute;

    /**
     * the perturbed solute.
     */
    Perturbed_Solute m_perturbed_solute;

    /**
     * is the atom perturbed?
     */
    std::vector<bool> m_is_perturbed;
    
    /**
     * the number of solvent molecules.
     */
    std::vector<unsigned int> m_num_solvent_molecules;
    
    /**
     * the number of solvent atoms.
     * vector for multiple solvents.
     */
    std::vector<unsigned int> m_num_solvent_atoms;
    
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
    std::vector<unsigned int> m_molecule;

    /**
     * the chargegroups.
     */
    std::vector<int> m_chargegroup;
        
    /**
     * the number of solute chargegroups.
     */
    unsigned int m_num_solute_chargegroups;
    
    /**
     * residue names (solute and solvent).
     */
    std::vector<std::string> m_residue_name;

    /**
     * energy groups.
     */
    std::vector<unsigned int> m_energy_group;
    
    /**
     * energy group of atom
     */
    std::vector<unsigned int> m_atom_energy_group;
    
    /**
     * lambda.
     */
    double m_lambda;
    
    /**
     * old lambda
     */
    double m_old_lambda;

    /**
     * lambda exponent
     */
    int m_lambda_exp;

    /**
     * interaction matrix for scaled interactions
     */
    std::map<std::pair<int, int>, std::pair<double, double> > 
    m_energy_group_scaling;

    /**
     * interaction matrix for interactions with
     * changed lambda dependence
     */
    std::map<std::pair<int, int>, std::pair<int, double> > 
    m_energy_group_lambdadep;

    /**
     * lambda primes for current lambda
     */
    std::vector<double> m_lambda_prime;
    
    /**
     * lambda prime / lambda derivatives
     */
    std::vector<double> m_lambda_prime_derivative;
    
    /**
     * alpha parameter values for the perturbed energy derivatives
     */
    std::vector<double> m_perturbed_energy_derivative_alpha;

    /**
     * position restraints / constraints
     */
    std::vector<position_restraint_struct> m_position_restraint;

    /**
     * jvalue restraints / constraints
     */
    std::vector<jvalue_restraint_struct> m_jvalue_restraint;

    /**
     * expanded topology for multiple unit cell simulations
     */
    Topology * m_multicell_topo;
    
    
  }; // topology
  
} // topology

#endif

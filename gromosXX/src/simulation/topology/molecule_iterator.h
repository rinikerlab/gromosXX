/**
 * @file molecule_iterator.h
 * iterator over the molecules.
 */

#ifndef INCLUDED_MOLECULE_ITERATOR
#define INCLUDED_MOLECULE_ITERATOR

namespace simulation
{
  /**
   * @class Molecule_Iterator
   * iterator over the molecules (SUBMOLECULES block)
   */
  class Molecule_Iterator 
    : public Atom_Group_Iterator<std::vector<size_t>::const_iterator>
  {
  public:
    Molecule_Iterator(std::vector<size_t>::const_iterator mol_it)
      : Atom_Group_Iterator<std::vector<size_t>::const_iterator>(mol_it) {};
    
    /**
     * calculates the molecular center of mass and molecular kinetic energy.
     * @param sys the system: used for atom positions, velocities and masses
     * and the periodicity to gather the molecule.
     * @param mass the masses of the atoms.
     * @param com_pos returns the calculated center of mass.
     * @param com_e_kin returns the molecular kinetic energy (vector).
     * @TODO the gathering of the molecule is hardcoded in here.
     * Maybe this should be changed to a generic implementation.
     * Gathering is done in respect to the previous atom. An idea would
     * be to gather as default with respect to the previous atom but
     * letting the user override this (GATHER block).
     * This does not yield the same answer as Phils approach for all cases
     * but maybe for the practical ones???
     */
    template<math::boundary_enum b>
    void com(simulation::System<b> const & sys, math::SArray const &mass,
	     math::Vec &com_pos, math::Matrix &com_e_kin)
    {
      com_pos = 0.0;

      double m;
      double tot_mass = 0.0;

      math::Vec p;
      math::Vec prev;
      math::Vec v = 0.0;

      prev = sys.pos()(*begin());
      // std::cout << "center of mass: " << *begin() << " - " << *end() << std::endl;
      for(Atom_Iterator it=begin(), to=end(); it!=to; ++it){
	// std::cout << *it << std::endl;
	
	assert(unsigned(mass.size()) > *it && unsigned(sys.pos().size()) > *it);

	m = mass(*it);
	tot_mass += m;
	sys.periodicity().nearest_image(sys.pos()(*it), prev, p);
	com_pos += m * (p + prev);
	v += m * sys.vel()(*it);
	prev += p;
      }
      com_pos /= tot_mass;
      
      for(int i=0; i<3; ++i)
	for(int j=0; j<3; ++j)
	  com_e_kin(i,j) = 0.5 * v(i) * v(j) / tot_mass;
    }

  private:
  };
  
} // simulation

#endif

  

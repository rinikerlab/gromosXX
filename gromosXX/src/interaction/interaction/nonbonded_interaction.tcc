/**
 * @file nonbonded_interaction.tcc
 * template methods of Nonbonded_Interaction.
 */

#undef MODULE
#undef SUBMODULE

#define MODULE interaction
#define SUBMODULE interaction

#include "../../debug.h"

/**
 * Constructor.
 */
template<typename t_simulation, typename t_pairlist>
inline interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::Nonbonded_Interaction()
  : Interaction<t_simulation>("NonBonded")
{
}

/**
 * Destructor.
 */
template<typename t_simulation, typename t_pairlist>
inline interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::~Nonbonded_Interaction()
{
  DEBUG(4, "Nonbonded_Interaction::destructor");
}

/**
 * add a lj parameter struct.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::add_lj_parameter(size_t iac_i, size_t iac_j, lj_parameter_struct lj)
{
  DEBUG(15, "Nonbonded_Interaction::add_lj_parameter " 
	<< iac_i << "-" << iac_j);
  
  assert(iac_i < m_lj_parameter.size());
  assert(iac_j < m_lj_parameter.size());
  assert(iac_i < m_lj_parameter[iac_j].size());
  assert(iac_j < m_lj_parameter[iac_i].size());
  
  m_lj_parameter[iac_i][iac_j] = lj;
  m_lj_parameter[iac_j][iac_i] = lj;
}

/** 
 * set the coulomb constant 
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::coulomb_constant(double const coulomb_constant)
{
  m_coulomb_constant = coulomb_constant;
}

/**
 * get the coulomb constant
 */
template<typename t_simulation, typename t_pairlist>
inline double interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::coulomb_constant()const
{
  return m_coulomb_constant;
}

/**
 * get the lj parameter for atom types iac_i, iac_j
 */
template<typename t_simulation, typename t_pairlist>
inline interaction::lj_parameter_struct const &
interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::lj_parameter(size_t iac_i, size_t iac_j)
{
  DEBUG(15, "Nonbonded_Interaction::get_lj_parameter " 
	<< iac_i << "-" << iac_j);

  assert(iac_i < m_lj_parameter.size());
  assert(iac_j < m_lj_parameter[iac_i].size());
  
  return m_lj_parameter[iac_i][iac_j];
}

/**
 * resize the matrix.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::resize(size_t i)
{
  m_lj_parameter.resize(i);
  typename std::vector< std::vector<lj_parameter_struct> >::iterator
    it = m_lj_parameter.begin(),
    to = m_lj_parameter.end();
  
  for(; it!=to; ++it)
    it->resize(i);
}

/**
 * calculate nonbonded forces and energies.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::calculate_interactions(t_simulation &sim)
{
  DEBUG(4, "Nonbonded_Interaction::calculate_interactions");

  // initialize the constants
  if (!sim.steps())
    initialize(sim);

  // need to update pairlist?
  DEBUG(7, "steps " << sim.steps() << " upd " << sim.nonbonded().update());

  if(!(sim.steps() % sim.nonbonded().update())){
    // create a pairlist
    DEBUG(7, "\tupdate the pairlist");
    m_pairlist.update(sim);
  
    // recalc long-range forces
    DEBUG(7, "\tlong range");
    m_longrange_force.resize(sim.system().force().size());
    m_longrange_force = 0.0;

    m_longrange_energy.resize(sim.system().energies().bond_energy.size());
    
    do_interactions(sim, m_pairlist.long_range().begin(),
		    m_pairlist.long_range().end(),
		    longrange);
  }

  // calculate forces / energies
  DEBUG(7, "\tshort range");
  do_interactions(sim, m_pairlist.short_range().begin(),
		  m_pairlist.short_range().end(),
		  shortrange);

  // add long-range force
  sim.system().force() += m_longrange_force;
  
  // and long-range energies
  for(size_t i = 0; i < m_longrange_energy.lj_energy.size(); ++i){
    for(size_t j = 0; j < m_longrange_energy.lj_energy.size(); ++j){
      sim.system().energies().lj_energy[i][j] += 
	m_longrange_energy.lj_energy[i][j];
      sim.system().energies().crf_energy[i][j] += 
	m_longrange_energy.crf_energy[i][j];
    }
  }

  // add 1,4 - interactions
  do_14_interactions(sim);

  // possibly do the RF contributions due to excluded atoms
  if(sim.nonbonded().RF_exclusion())
    do_RF_excluded_interactions(sim);

}

/**
 * helper function to initialize the constants.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::initialize(t_simulation const &sim)
{
  // Force
  m_cut3i = 
    1.0 / ( sim.nonbonded().RF_cutoff() 
	    * sim.nonbonded().RF_cutoff() 
	    * sim.nonbonded().RF_cutoff());

  m_crf_cut3i = sim.nonbonded().RF_constant() * m_cut3i;

  // Energy
  m_crf_2cut3i = sim.nonbonded().RF_constant() / 2.0 * m_cut3i;

  m_crf_cut = (1 - sim.nonbonded().RF_constant() / 2.0)
    / sim.nonbonded().RF_cutoff();

}


/**
 * helper function to calculate forces and energies, 
 * stores them in the arrays pointed to by parameters
 * to make it usable for longrange calculations.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::do_interactions(t_simulation &sim, typename t_pairlist::iterator it, 
		  typename t_pairlist::iterator to, 
		  nonbonded_type_enum range)
{
  math::Vec r, f;
  double e_lj, e_crf;
  
  math::VArray &pos = sim.system().pos();

  math::VArray *force;
  simulation::Energy *energy;
  
  if (range == shortrange){
    force = &sim.system().force();
    energy = &sim.system().energies();
  }
  else{
    force = &m_longrange_force;
    energy = &m_longrange_energy;
  }
  
  DEBUG(7, "\tcalculate interactions");  

  for( ; it != to; ++it){
    
    DEBUG(10, "\tpair\t" << it.i() << "\t" << *it);

    sim.system().periodicity().nearest_image(pos(it.i()), pos(*it), r);

    const lj_parameter_struct &lj = 
      lj_parameter(sim.topology().iac(it.i()),
		   sim.topology().iac(*it));

    DEBUG(11, "\tlj-parameter c6=" << lj.c6 << " c12=" << lj.c12);

    lj_crf_interaction(r, lj.c6, lj.c12,
		       sim.topology().charge()(it.i()) * 
		       sim.topology().charge()(*it),
		       f, e_lj, e_crf);

    (*force)(it.i()) += f;
    (*force)(*it) -= f;

    // energy
    (*energy).lj_energy[sim.topology().atom_energy_group(it.i())]
      [sim.topology().atom_energy_group(*it)] += e_lj;

    (*energy).crf_energy[sim.topology().atom_energy_group(it.i())]
      [sim.topology().atom_energy_group(*it)] += e_crf;

    DEBUG(11, "\ti and j " << sim.topology().atom_energy_group(it.i())
	  << " " << sim.topology().atom_energy_group(*it));
    
  }
  
}

/**
 * helper function to calculate the force and energy for
 * a given atom pair.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::lj_crf_interaction(math::Vec const &r,
		     double const c6, double const c12,
		     double const q,
		     math::Vec &force, double &e_lj, double &e_crf)
{
  assert(dot(r,r) != 0);
  const double dist2 = dot(r, r);
  const double dist2i = 1.0 / dist2;
  const double dist6i = dist2i * dist2i * dist2i;
  const double disti = sqrt(dist2i);
  
  force = ((2 * c12 * dist6i - c6) * 6.0 * dist6i * dist2i + 
    q * coulomb_constant() * (disti * dist2i + m_crf_cut3i)) * r;

  e_lj = (c12 * dist6i - c6) * dist6i;
  e_crf = q * coulomb_constant() * (disti - m_crf_2cut3i * dist2 - m_crf_cut);
  
}

/**
 * helper function to calculate the force and energy for
 * the reaction field contribution for a given pair
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::rf_interaction(math::Vec const &r,double const q,
		 math::Vec &force, double &e_crf)
{
  const double dist2 = dot(r, r);
  
  force = q * coulomb_constant() *  m_crf_cut3i * r;

  e_crf = q * coulomb_constant() * ( -m_crf_2cut3i * dist2 - m_crf_cut);
  DEBUG(11, "dist2 " << dist2 );
  DEBUG(11, "crf_2cut3i " << m_crf_2cut3i);
  DEBUG(11, "crf_cut " << m_crf_cut);
  DEBUG(11, "q*q   " << q );
  
}

/**
 * helper function to calculate the forces and energies from the
 * 1,4 interactions.
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::do_14_interactions(t_simulation &sim)
{
  math::Vec r, f;
  double e_lj, e_crf;

  math::VArray &pos   = sim.system().pos();
  math::VArray &force = sim.system().force();
  
  DEBUG(7, "\tcalculate 1,4-interactions");

  std::set<int>::const_iterator it, to;
  
  for(size_t i=0; i<sim.topology().num_solute_atoms(); ++i){
    it = sim.topology().one_four_pair(i).begin();
    to = sim.topology().one_four_pair(i).end();
    
    for( ; it != to; ++it){
      DEBUG(11, "\tpair " << i << " - " << *it);
      
      sim.system().periodicity().nearest_image(pos(i), pos(*it), r);

      const lj_parameter_struct &lj = 
	lj_parameter(sim.topology().iac(i),
		     sim.topology().iac(*it));

      DEBUG(11, "\tlj-parameter cs6=" << lj.cs6 << " cs12=" << lj.cs12);

      lj_crf_interaction(r, lj.cs6, lj.cs12,
			 sim.topology().charge()(i) * 
			 sim.topology().charge()(*it),
			 f, e_lj, e_crf);

      force(i) += f;
      force(*it) -= f;

    // energy
    sim.system().energies().lj_energy[sim.topology().atom_energy_group(i)]
      [sim.topology().atom_energy_group(*it)] += e_lj;

    sim.system().energies().crf_energy[sim.topology().atom_energy_group(i)]
      [sim.topology().atom_energy_group(*it)] += e_crf;
    DEBUG(11, "i and j from 1,4" << sim.topology().atom_energy_group(i) << ", " << sim.topology().atom_energy_group(*it));
    

    } // loop over 1,4 pairs
  } // loop over solute atoms
}  

/**
 * helper function to calculate the forces and energies from the
 * RF contribution of excluded atoms and self term
 */
template<typename t_simulation, typename t_pairlist>
inline void interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::do_RF_excluded_interactions(t_simulation &sim)
{
  math::Vec r, f;
  double e_crf;
  std::cout.precision(10);
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  
  math::VArray &pos   = sim.system().pos();
  math::VArray &force = sim.system().force();
  
  DEBUG(7, "\tcalculate RF excluded interactions");

  std::set<int>::const_iterator it, to;
  
  for(size_t i=0; i<sim.topology().num_solute_atoms(); ++i){
    it = sim.topology().exclusion(i).begin();
    to = sim.topology().exclusion(i).end();
    
    DEBUG(11, "\tself-term " << i );
    r=0;
    
    // this will only contribute in the energy, the force should be zero.
    rf_interaction(r,sim.topology().charge()(i) * sim.topology().charge()(i),
		   f, e_crf);
    sim.system().energies().crf_energy[sim.topology().atom_energy_group(i)]
      [sim.topology().atom_energy_group(i)] += 0.5 * e_crf;
    DEBUG(11, "\tcontribution " << 0.5*e_crf);
    
    for( ; it != to; ++it){
      DEBUG(11, "\tpair " << i << " - " << *it);
      
      sim.system().periodicity().nearest_image(pos(i), pos(*it), r);


      rf_interaction(r, sim.topology().charge()(i) * 
		     sim.topology().charge()(*it),
		     f, e_crf);

      force(i) += f;
      force(*it) -= f;

      // energy
      sim.system().energies().crf_energy[sim.topology().atom_energy_group(i)]
      [sim.topology().atom_energy_group(*it)] += e_crf;
      DEBUG(11, "\tcontribution " << e_crf);
      
    } // loop over excluded pairs
  } // loop over solute atoms
  // Solvent
  simulation::chargegroup_iterator cg_it = sim.topology().chargegroup_begin(),
    cg_to = sim.topology().chargegroup_end();
  cg_it += sim.topology().num_solute_chargegroups();
  
  for( ; cg_it != cg_to; ++cg_it){

    // loop over the atoms
    simulation::Atom_Iterator at_it = cg_it.begin(),
      at_to = cg_it.end();

    for ( ; at_it != at_to; ++at_it){
      DEBUG(11, "\tsolvent self term " << *at_it);
      // no solvent self term. The distance dependent part and the forces
      // are zero. The distance independent part should add up to zero 
      // for the energies and is left out.

      for(simulation::Atom_Iterator at2_it=at_it+1; at2_it!=at_to; ++at2_it){
	
	DEBUG(11, "\tsolvent " << *at_it << " - " << *at2_it);
	sim.system().periodicity().nearest_image(pos(*at_it), 
						 pos(*at2_it), r);

	// for solvent, we don't calculate internal forces (rigid molecules)
	// and the distance independent parts should go to zero
	e_crf = -sim.topology().charge()(*at_it) * 
	  sim.topology().charge()(*at2_it) * coulomb_constant() * 
	  m_crf_2cut3i * dot(r,r);
	
	// energy
	sim.system().energies().crf_energy
	  [sim.topology().atom_energy_group(*at_it) ]
	  [sim.topology().atom_energy_group(*at2_it)] += e_crf;
      } // loop over at2_it
    } // loop over at_it
  } // loop over solvent charge groups
}  

/**
 * pairlist accessor
 */
template<typename t_simulation, typename t_pairlist>
t_pairlist & 
interaction::Nonbonded_Interaction<t_simulation, t_pairlist>
::pairlist()
{
  return m_pairlist;
}

  

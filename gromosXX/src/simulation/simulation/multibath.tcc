/**
 * @file multibath.tcc
 * methods for the multibath parameter class.
 */

#undef MODULE
#undef SUBMODULE
#define MODULE simulation
#define SUBMODULE simulation

#include "../../debug.h"

inline
simulation::Multibath::Multibath()
{
}

inline void simulation::Multibath
::add_bath(simulation::bath_struct s)
{
  push_back(s);
}

inline void simulation::Multibath
::add_bath(int last_atom, double temperature, double tau,
	   double dof, double solute_constr_dof, double solvent_constr_dof)
{
  bath_struct s;
  s.last_atom = last_atom;
  s.temperature = temperature;
  s.tau = tau;
  s.dof = dof;
  s.solute_constr_dof = solute_constr_dof;
  s.solvent_constr_dof = solvent_constr_dof;
  s.kinetic_energy = 0;
  
  add_bath(s);
}

inline simulation::bath_struct & 
simulation::Multibath::bath(size_t i)
{
  assert(i < size());
  return (*this)[i];  
}

inline simulation::bath_struct const & 
simulation::Multibath::bath(size_t i)const
{
  assert(i < size());
  return (*this)[i];  
}

inline int simulation::Multibath::in_bath(size_t const i)const
{
  std::vector<bath_struct>::const_iterator it = begin(),
    to = end();
  
  for(size_t b=0; it != to; ++it, ++b)
    if (it->last_atom >= i) return b;
  
  return -1;
}

template<typename t_topology>
inline void simulation::Multibath
::calculate_degrees_of_freedom(t_topology &topo)
{
  // check whether the last last is really the last_atom
  if (size() == 0 || 
      (end()-1)->last_atom != topo.num_atoms()-1){

    io::messages.add("Adding a bath, no temperature coupling",
		     "Multibath::calculate_degrees_of_freedom",
		     io::message::notice);
    add_bath(topo.num_atoms()-1, 0.0, -1);
    
  }

  std::vector<bath_struct>::iterator it = begin(),
    to = end();

  // loop over the baths
  for(size_t last=0; it != to; ++it){
    it->dof = (it->last_atom - last + 1) * 3;
    last = it->last_atom + 1;
  }

  // substract constraints
  std::vector<compound::distance_constraint_struct>::const_iterator 
    c_it = topo.solute().distance_constraints().begin(),
    c_to = topo.solute().distance_constraints().end();
  
  for( ; c_it != c_to; ++c_it){
    
    DEBUG(10, "Constraint: " << c_it->i << " - " << c_it->j);

    (*this)[in_bath(c_it->i)].dof -= 0.5;
    (*this)[in_bath(c_it->j)].dof -= 0.5;

    (*this)[in_bath(c_it->i)].solute_constr_dof += 0.5;
    (*this)[in_bath(c_it->j)].solute_constr_dof += 0.5;

  }
  
  for(size_t i=0; i<size(); ++i){
    DEBUG(7, "dof           " << (*this)[i].dof);
    DEBUG(7, "solute constr " << (*this)[i].solute_constr_dof);
  }

  // solvent constraints
  int index = topo.num_solute_atoms();
  for(size_t s=0; s < topo.num_solvents(); ++s){
    
    for(size_t m=0; m < topo.num_solvent_molecules(s); ++m){

      c_it = topo.solvent(s).distance_constraints().begin();
      c_to = topo.solvent(s).distance_constraints().end();
      
      for( ; c_it != c_to; ++c_it){
	
	(*this)[in_bath(c_it->i + index)].dof -= 0.5;
	(*this)[in_bath(c_it->j + index)].dof -= 0.5;
	
	(*this)[in_bath(c_it->i + index)].solvent_constr_dof += 0.5;
	(*this)[in_bath(c_it->j + index)].solvent_constr_dof += 0.5;
	
      }
      
      index += topo.solvent(s).num_atoms();
      
    }
    
  }
  
}

inline int simulation::Multibath::check_state(size_t const num_atoms)const
{
  int result = 0;
  size_t last_atom = 0;
  std::vector<bath_struct>::const_iterator it = begin(),
    to = end();
  for( ; it!=to; ++it){
    if (it->last_atom < last_atom){
      io::messages.add("Multibath not sorted", "Multibath::check_state",
		       io::message::error);
      ++result;
    }
    if (it->last_atom > num_atoms){
      io::messages.add("Multibath last atom index too large",
		       "Multibath::check_state",
		       io::message::error);
      ++result;
    }
    if (it->dof == 0)
      io::messages.add("Multibath: bath with 0 degrees of freedom?",
		       "Multibath::check_state",
		       io::message::warning);
    if (it->solute_constr_dof < 0 || it->solvent_constr_dof < 0){
      io::messages.add("Multibath: constrained degrees of freedom negative",
		       "Multibath::check_state",
		       io::message::error);
      ++result;
    }
    if (it->tau < 0 && it->tau != -1){
      io::messages.add("Multibath: tau < 0 && tau != -1",
		       "Multibath::check_state",
		       io::message::error);
      ++result;
    }
    if (it->temperature < 0){
      io::messages.add("Multibath: temperature < 0",
		       "Multibath::check_state",
		       io::message::error);
      ++result;
    }
    
  }
  
  return result;

}

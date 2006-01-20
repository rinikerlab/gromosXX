/**
 * @file shake.cc
 * contains the template methods for
 * the class Shake.
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <interaction/interaction.h>
#include <interaction/interaction_types.h>

#include <math/periodicity.h>

#include <algorithm/constraints/shake.h>

#include <util/template_split.h>
#include <util/error.h>
#include <util/debug.h>

#undef MODULE
#undef SUBMODULE
#define MODULE algorithm
#define SUBMODULE constraints

// dihedral constraints template method
#include "dihedral_constraint.cc"

/**
 * Constructor.
 */
algorithm::Shake
::Shake(double const tolerance, int const max_iterations,
	std::string const name)
  : Algorithm(name),
    m_tolerance(tolerance),
    m_max_iterations(max_iterations),
    m_solvent_timing(0.0)
{
}

/**
 * Destructor.
 */
algorithm::Shake
::~Shake()
{
}

void algorithm::Shake
::tolerance(double const tol)
{
  m_tolerance = tol;
}

/**
 * do one iteration
 */
template<math::boundary_enum B, math::virial_enum V>
int algorithm::Shake::shake_iteration
(
 topology::Topology const &topo,
 configuration::Configuration & conf,
 bool & convergence,
 int first,
 std::vector<bool> &skip_now,
 std::vector<bool> &skip_next,
 std::vector<topology::two_body_term_struct> const & constr,
 double dt,
 math::Periodicity<B> const & periodicity
 )

{
  convergence = true;

  // index for constraint_force...
  unsigned int k = 0;
  double const dt2 = dt * dt;
  
  // and constraints
  for(typename std::vector<topology::two_body_term_struct>
	::const_iterator
	it = constr.begin(),
	to = constr.end();
      it != to;
      ++it, ++k ){
	
    // check whether we can skip this constraint
    if (skip_now[it->i] && skip_now[it->j]) continue;

    DEBUG(10, "i: " << it->i << " j: " << it->j << " first: " << first);

    // the position
    math::Vec &pos_i = conf.current().pos(first+it->i);
    math::Vec &pos_j = conf.current().pos(first+it->j);

    DEBUG(10, "\ni: " << math::v2s(pos_i) << "\nj: " << math::v2s(pos_j));
	
    math::Vec r;
    periodicity.nearest_image(pos_i, pos_j, r);
    DEBUG(12, "ni:  " << math::v2s(r));
    
    double dist2 = abs2(r);
	
    assert(parameter().size() > it->type);
    double constr_length2 = parameter()[it->type].r0 * parameter()[it->type].r0;
    double diff = constr_length2 - dist2;

    DEBUG(13, "constr: " << constr_length2 << " dist2: " << dist2);
	  
    if(fabs(diff) >= constr_length2 * tolerance() * 2.0){
      // we have to shake
      DEBUG(10, "shaking");
      
      // the reference position
      const math::Vec &ref_i = conf.old().pos(first+it->i);
      const math::Vec &ref_j = conf.old().pos(first+it->j);
      
      math::Vec ref_r;
      periodicity.nearest_image(ref_i, ref_j, ref_r);

      double sp = dot(ref_r, r);
	  
      if(sp < constr_length2 * math::epsilon){
	/*
	io::messages.add("SHAKE error. vectors orthogonal",
			 "Shake::???",
			 io::message::critical);
	*/
	DEBUG(5, "ref i " << math::v2s(ref_i) << " ref j " << math::v2s(ref_j));
	DEBUG(5, "free i " << math::v2s(pos_i) << " free j " << math::v2s(pos_j));
	DEBUG(5, "ref r " << math::v2s(ref_r));
	DEBUG(5, "r " << math::v2s(r));

	std::cout << "SHAKE ERROR\n"
		  << "\tatom i    : " << first + it->i + 1 << "\n"
		  << "\tatom j    : " << first + it->j + 1 << "\n"
	  // << "\tfirst     : " << first << "\n"
		  << "\tref i     : " << math::v2s(ref_i) << "\n"
		  << "\tref j     : " << math::v2s(ref_j) << "\n"
		  << "\tfree i    : " << math::v2s(pos_i) << "\n"
		  << "\tfree j    : " << math::v2s(pos_j) << "\n"
		  << "\tref r     : " << math::v2s(ref_r) << "\n"
		  << "\tr         : " << math::v2s(r) << "\n"
		  << "\tsp        : " << sp << "\n"
		  << "\tconstr    : " << constr_length2 << "\n"
		  << "\tdiff      : " << diff << "\n"
		  << "\tforce i   : " << math::v2s(conf.old().force(first+it->i)) << "\n"
		  << "\tforce j   : " << math::v2s(conf.old().force(first+it->j)) << "\n"
		  << "\tvel i     : " << math::v2s(conf.current().vel(first+it->i)) << "\n"
		  << "\tvel j     : " << math::v2s(conf.current().vel(first+it->j)) << "\n"
		  << "\told vel i : " << math::v2s(conf.old().vel(first+it->i)) << "\n"
		  << "\told vel j : " << math::v2s(conf.old().vel(first+it->j)) << "\n\n";
	
	return E_SHAKE_FAILURE;
      }
	  
      // lagrange multiplier
      double lambda = diff / (sp * 2 *
			      (1.0 / topo.mass()(first+it->i) +
			       1.0 / topo.mass()(first+it->j) ));      

      DEBUG(10, "lagrange multiplier " << lambda);

      /*
      if (do_constraint_force == true){
	
	//if it is a solute sum up constraint forces
	assert(unsigned(sys.constraint_force().size()) > k + force_offset);
	sys.constraint_force()(k+force_offset) += (lambda * ref_r);
	// m_lambda(k) += lambda;
      }
      */

      if (V == math::atomic_virial){
	for(int a=0; a<3; ++a){
	  for(int aa=0; aa<3; ++aa){
	    conf.old().virial_tensor(a,aa) +=
	      ref_r(a) * ref_r(aa) * lambda / dt2;
	  }
	}
	DEBUG(12, "\tatomic virial done");
      }
      
      // update positions
      ref_r *= lambda;
      pos_i += ref_r / topo.mass()(first+it->i);
      pos_j -= ref_r / topo.mass()(first+it->j);
	  
      convergence = false;

      // consider atoms in the next step
      skip_next[it->i] = false;
      skip_next[it->j] = false;
      
    } // we have to shake
  } // constraints
      
  
  return 0;

}    

/**
 * shake solute
 */
template<math::boundary_enum B, math::virial_enum V>
void algorithm::Shake::
solute(topology::Topology const & topo,
       configuration::Configuration & conf,
       simulation::Simulation const & sim,
       int const max_iterations,
       int & error)
{
  // for now shake the whole solute in one go,
  // not bothering about submolecules...

  DEBUG(8, "\tshaking SOLUTE");
  math::Periodicity<B> periodicity(conf.current().box);
  
  // conf.constraint_force() = 0.0;
  // m_lambda = 0.0;

  const double start = util::now();
  
  std::vector<bool> skip_now;
  std::vector<bool> skip_next;
  int num_iterations = 0;

  int first = 0;

  skip_now.assign(topo.solute().num_atoms(), false);
  skip_next.assign(topo.solute().num_atoms(), true);
  
  bool convergence = false;
  while(!convergence){
    DEBUG(9, "\titeration" << std::setw(10) << num_iterations);

    // distance constraints
    bool dist_convergence = true;

    if (topo.solute().distance_constraints().size() && 
	sim.param().constraint.solute.algorithm == simulation::constr_shake &&
	sim.param().constraint.ntc > 1){

      DEBUG(7, "SHAKE: distance constraints iteration");

      if(shake_iteration<B, V>
	 (topo, conf, dist_convergence, first, skip_now, skip_next,
	  topo.solute().distance_constraints(), sim.time_step_size(),
	  periodicity)
	 ){
	io::messages.add("SHAKE error. vectors orthogonal",
			 "Shake::solute",
			 io::message::error);
	std::cout << "SHAKE failure in solute!" << std::endl;
	error = E_SHAKE_FAILURE_SOLUTE;
	return;
      }
    }

    // dihedral constraints
    bool dih_convergence = true;
    if (sim.param().dihrest.dihrest == 3){

      DEBUG(7, "SHAKE: dihedral constraints iteration");
      
      if(dih_constr_iteration<B, V>
	 (topo, conf, sim, dih_convergence, skip_now, skip_next, periodicity)
	 ){
	io::messages.add("SHAKE error: dihedral constraints",
			 "Shake::solute",
			 io::message::error);
	std::cout << "SHAKE failure in solute dihedral constraints!" << std::endl;
	error = E_SHAKE_FAILURE_SOLUTE;
	return;
      }
    }

    convergence = dist_convergence && dih_convergence;
    
    if(++num_iterations > max_iterations){
      io::messages.add("SHAKE error. too many iterations",
		       "Shake::solute",
		       io::message::error);
      error = E_SHAKE_FAILURE_SOLUTE;
      return;
    }

    skip_now = skip_next;
    skip_next.assign(skip_next.size(), true);

  } // convergence?

  // constraint_force
  /*
  for (unsigned int i=0; i < topo.solute().distance_constraints().size();++i){
    conf.constraint_force()(i) *= 1 /(dt * dt);
    DEBUG(5, "constraint_force " << sqrt(abs2(conf.constraint_force()(i)) ));
  }
  */

  m_timing += util::now() - start;

  error = 0;

} // solute


/**
 * shake solvent.
 */
template<math::boundary_enum B, math::virial_enum V>
void algorithm::Shake
::solvent(topology::Topology const & topo,
	  configuration::Configuration & conf,
	  double dt, int const max_iterations,
	  int & error)
{

  DEBUG(8, "\tshaking SOLVENT");
  
  const double start = util::now();

  // the first atom of a solvent
  unsigned int first = topo.num_solute_atoms();

  std::vector<bool> skip_now;
  std::vector<bool> skip_next;
  int tot_iterations = 0;

  math::Periodicity<B> periodicity(conf.current().box);

  // for all solvents
  for(unsigned int i=0; i<topo.num_solvents(); ++i){

    // loop over the molecules
    for(unsigned int nm=0; nm<topo.num_solvent_molecules(i);
	++nm, first+=topo.solvent(i).num_atoms()){

      skip_now.assign(topo.solvent(i).num_atoms(), false);
      skip_next.assign(topo.solvent(i).num_atoms(), true);

      int num_iterations = 0;
      bool convergence = false;
      while(!convergence){
	DEBUG(9, "\titeration" << std::setw(10) << num_iterations);

	if(shake_iteration<B, V>
	   (topo, conf, convergence, first, skip_now, skip_next,
	    topo.solvent(i).distance_constraints(), dt,
	    periodicity)){
	  
	  io::messages.add("SHAKE error. vectors orthogonal",
			   "Shake::solvent", io::message::error);
	  
	  std::cout << "SHAKE failure in solvent!" << std::endl;
	  error = E_SHAKE_FAILURE_SOLVENT;
	  return;
	}
	
	// std::cout << num_iterations+1 << std::endl;
	if(++num_iterations > max_iterations){
	  io::messages.add("SHAKE error. too many iterations",
			   "Shake::solvent",
			   io::message::critical);
	  error = E_SHAKE_FAILURE_SOLVENT;
	  return;
	}

	skip_now = skip_next;
	skip_next.assign(skip_next.size(), true);

      } // while(!convergence)
      
      tot_iterations += num_iterations;
      
    } // molecules
    
  } // solvents

  m_solvent_timing += util::now() - start;
  DEBUG(3, "total shake solvent iterations: " << tot_iterations);
  error = 0;
} // shake solvent

/**
 * apply the SHAKE algorithm
 */
int algorithm::Shake::apply(topology::Topology & topo,
			    configuration::Configuration & conf,
			    simulation::Simulation & sim)
{
  DEBUG(7, "applying SHAKE");
  bool do_vel_solute = false;
  bool do_vel_solvent = false;
  
  int error = 0;
  
  // check whether we shake
  if ((topo.solute().distance_constraints().size() && 
       sim.param().constraint.solute.algorithm == simulation::constr_shake &&
       sim.param().constraint.ntc > 1) ||
      sim.param().dihrest.dihrest == 3){
    
    DEBUG(8, "\twe need to shake SOLUTE");

    do_vel_solute = true;

    SPLIT_VIRIAL_BOUNDARY(solute,
			  topo, conf, sim, 
			  m_max_iterations, error);

    if (error){
      std::cout << "SHAKE: exiting with error condition: E_SHAKE_FAILURE_SOLUTE "
		<< "at step " << sim.steps() << std::endl;
      // save old positions to final configuration... (even before free-flight!)
      conf.current().pos = conf.old().pos;
      return E_SHAKE_FAILURE_SOLUTE;
    }
  }
  
  if (sim.param().system.nsm &&
      sim.param().constraint.solvent.algorithm == simulation::constr_shake){

    DEBUG(8, "\twe need to shake SOLVENT");
    do_vel_solvent = true;

    SPLIT_VIRIAL_BOUNDARY(solvent, 
			  topo, conf, sim.time_step_size(), 
			  m_max_iterations, error);
    if (error){
      std::cout << "SHAKE: exiting with error condition: E_SHAKE_FAILURE_SOLVENT "
		<< "at step " << sim.steps() << std::endl;
      // save old positions to final configuration... (even before free-flight!)
      conf.current().pos = conf.old().pos;
      return E_SHAKE_FAILURE_SOLVENT;
    }
  }
  
  // shaken velocity:
  // stochastic dynamics needs to shake without velocity correction
  // (once; it shakes twice...)
  if (!sim.param().stochastic.sd){
    if (do_vel_solute){
      for(unsigned int i=0; i<topo.num_solute_atoms(); ++i)
	conf.current().vel(i) = (conf.current().pos(i) - conf.old().pos(i)) / 
	  sim.time_step_size();
    }
    if (do_vel_solvent){
      for(unsigned int i=topo.num_solute_atoms(); i < topo.num_atoms(); ++i)
	conf.current().vel(i) = (conf.current().pos(i) - conf.old().pos(i)) / 
	  sim.time_step_size();
    }
  }
  
  // return success!
  return 0;
		   
}

int algorithm::Shake::init(topology::Topology & topo,
			   configuration::Configuration & conf,
			   simulation::Simulation & sim,
			   std::ostream & os,
			   bool quiet)
{
  if (!quiet){
    os << "SHAKE\n"
	      << "\tsolute\t";
    if (sim.param().constraint.solute.algorithm == simulation::constr_shake){    
      os << "ON\n";  
      os << "\t\ttolerance = "
		<< sim.param().constraint.solute.shake_tolerance << "\n";
    }
    else os << "OFF\n";
  
    os << "\tsolvent\t";
  
    if (sim.param().constraint.solvent.algorithm == simulation::constr_shake){
      os << "ON\n";
      os << "\t\ttolerance = " 
		<< sim.param().constraint.solvent.shake_tolerance << "\n";
    }  else os << "OFF\n";
  }
  
  if (sim.param().start.shake_pos){
    if (!quiet)
      os << "\n\tshaking initial positions\n";

    // old and current pos and vel are the same...
    conf.old().pos = conf.current().pos;
    conf.old().vel = conf.current().vel;

    // shake the current ones
    if (apply(topo, conf, sim))
      return E_SHAKE_FAILURE;

    // restore the velocities
    conf.current().vel = conf.old().vel;
    
    // take a step back
    conf.old().pos = conf.current().pos;
    
    if (sim.param().start.shake_vel){
      if (!quiet)
	os << "\tshaking initial velocities\n";

      for(unsigned int i=0; i<topo.num_atoms(); ++i)
      conf.current().pos(i) = conf.old().pos(i) - 
	sim.time_step_size() * conf.old().vel(i);
    
      // shake again
      if (apply(topo, conf, sim))
	return E_SHAKE_FAILURE;
    
      // restore the positions
      conf.current().pos = conf.old().pos;
    
      // velocities are in opposite direction (in time)
      for(unsigned int i=0; i<topo.num_atoms(); ++i)
	conf.current().vel(i) = -1.0 * conf.current().vel(i);
      conf.old().vel = conf.current().vel;
    }
    
  }
  else if (sim.param().start.shake_vel){
    io::messages.add("shaking velocities without shaking positions illegal.",
		     "shake", io::message::error);
  }
  
  if (!quiet)
    os << "END\n";
  
  return 0;
}

void algorithm::Shake::print_timing(std::ostream & os)
{
  os << "    "
     << std::setw(40) << std::left << "Shake::solute"
     << std::setw(20) << m_timing << "\n"
     << "    "
     << std::setw(40) << std::left << "Shake::solvent"
     << std::setw(20) << m_solvent_timing << "\n";
}

/**
 * @file in_perturbation.cc
 * implements methods of In_Perturbation.
 */

#include <util/stdheader.h>

#include <topology/core/core.h>

#include <topology/solute.h>
#include <topology/solvent.h>
#include <topology/perturbed_atom.h>
#include <topology/perturbed_solute.h>

#include <topology/topology.h>
#include <simulation/multibath.h>
#include <simulation/parameter.h>
#include <io/instream.h>

#include <io/blockinput.h>

#include "in_perturbation.h"


#undef MODULE
#undef SUBMODULE
#define MODULE io
#define SUBMODULE topology

/**
 * Constructor.
 */

io::In_Perturbation::In_Perturbation(std::istream &is) 
  : GInStream(is) 
{
  // read the whole file at beginning
  readStream();
};

template<typename T>
static bool check_type(std::vector<std::string> const & buffer, std::vector<T> term)
{
  if (buffer.size()){
    std::istringstream is(buffer[1]);
    int num;
    if (!(is >> num) || num < 0)
      return false;
      
    for(typename std::vector<T>::const_iterator
	  it = term.begin(),
	  to = term.end();
	it != to;
	++it){
      
      if (int(it->A_type) >= num && int(it->B_type) >= num)
	return false;
    }
  }
  else return false;
  return true;
}

void
io::In_Perturbation::read(topology::Topology &topo,
			  simulation::Parameter &param)
{
  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  DEBUG(7, "Reading PERTURBATION");
  std::cout << "PERTURBATION TOPOLOGY\n";

  std::cout << title << "\n";

  // prepare arrays
  topo.is_perturbed().resize(topo.num_solute_atoms(), false);

  { // PERTBOND03
    buffer = m_block["PERTBOND03"];
    if (buffer.size()){
      std::cout << "\tPERTBOND03\n";

      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;
      
      if (param.constraint.ntc == 2){
	io::messages.add("No perturbed distance constraints for "
			 "NTC = 2 from perturbed bonds",
			 "in_perturbation",
			 io::message::warning);
      }
      else if (param.constraint.ntc == 3){
	std::cout << "\n\t\t"
		  << num
		  << " perturbed bonds from PERTBOND03 block added to "
		  << "perturbed distance constraints.";
      }

      std::cout << "\n\t"
		<< std::setw(10) << "atom i"
		<< std::setw(10) << "atom j"
		<< std::setw(10) << "type A"
		<< std::setw(10) << "type B"
		<< "\n";
      
      for(n=0; it != buffer.end() -1; ++it, ++n){
	int i, j, t_A, t_B;
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> i >> j >> t_A >> t_B;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTBOND03 block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error("bad line in PERTBOND03 block");
	}
	
	topology::two_body_term_struct b(i-1, j-1, t_A-1);

	if (param.constraint.ntc != 3){
	  std::vector<topology::two_body_term_struct>::iterator b_it
	    = std::find(topo.solute().bonds().begin(), 
			topo.solute().bonds().end(), 
			b);
	
	  if (b_it == topo.solute().bonds().end()){
	    io::messages.add("Perturbation of a non-existing bond "
			     "in PERTBOND03 block.",
			     "InTopology", io::message::error);
	  }
	
	  topo.solute().bonds().erase(b_it);
	  topology::perturbed_two_body_term_struct 
	    pb(i-1, j-1, t_A-1, t_B-1);

	  std::cout << "\t" 
		    << std::setw(10) << pb.i+1 
		    << std::setw(10) << pb.j+1
		    << std::setw(10) << pb.A_type+1 
		    << std::setw(10) << pb.B_type+1 
		    << "\n";
	  
	  topo.perturbed_solute().bonds().push_back(pb);
	}
	else{
	  std::vector<topology::two_body_term_struct>::iterator b_it
	    = std::find(topo.solute().distance_constraints().begin(), 
			topo.solute().distance_constraints().end(), 
			b);
	  
	  if (b_it == topo.solute().distance_constraints().end()){
	    io::messages.add("Perturbation of a non-existing distance "
			     "constraint in PERTBOND03 block.",
			     "InTopology", io::message::error);
	  }
	
	  topo.solute().distance_constraints().erase(b_it);
	  topology::perturbed_two_body_term_struct 
	    pb(i-1, j-1, t_A-1, t_B-1);

	  std::cout << "\t" 
		    << std::setw(10) << pb.i+1 
		    << std::setw(10) << pb.j+1
		    << std::setw(10) << pb.A_type+1 
		    << std::setw(10) << pb.B_type+1 
		    << "\n";
	  
	  topo.perturbed_solute().distance_constraints().push_back(pb);

	}

      }
      
      if (n != num){
	  io::messages.add("Wrong number of bonds in PERTBOND03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in PERTBOND03 block (n != num)");
      }
      else if (_lineStream.fail()){
	io::messages.add("Bad line in PERTBOND03 block.",
			 "InTopology", io::message::error);
    	throw std::runtime_error("error in PERTBOND03 block (fail)");
      }

      std::cout << "\n\t\tbonds :                          " 
		<< topo.solute().bonds().size()
		<< "\n\t\tperturbed bonds :                "
		<< topo.perturbed_solute().bonds().size()
		<< "\n\t\tdistance constraints :           "
		<< topo.solute().distance_constraints().size()
		<< "\n\t\tperturbed distance constraints : "
		<< topo.perturbed_solute().distance_constraints().size()
		<< "\n\n";
      
      std::cout << "\tEND\n";

    } // if block present
    
  } // PERTBOND03

  { // PERTCONSTRAINT03
    DEBUG(10, "PERTCONSTRAINT03 block");
    buffer = m_block["PERTCONSTRAINT03"];
  
    if (buffer.size() && param.constraint.ntc != 1){
      
      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;

      std::cout << "\tPERTCONSTRAINT03\n\t\t"
		<< num
		<< " bonds in PERTCONSTRAINT03 block."
		<< "\n\t\ttotal of perturbed constraint bonds : " 
		<< num + topo.perturbed_solute().distance_constraints().size()
		<< "\n";
  
      std::cout << "\t"
		<< std::setw(10) << "atom i"
		<< std::setw(10) << "atom j"
		<< std::setw(10) << "type A"
		<< std::setw(10) << "type B"
		<< "\n";
    
      for(n=0; it != buffer.end() - 1; ++it, ++n){
	int i, j, t_A, t_B;
	
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> i >> j >> t_A >> t_B;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTCONSTRAINT03 block",
			   "In_Topology", io::message::error);
	  throw std::runtime_error("bad line in PERTCONSTRAINT03 block");
	}
      
	if (i > int(topo.num_solute_atoms()) || 
	    j > int(topo.num_solute_atoms()) ||
	    i < 1 || j < 1){
	  io::messages.add("Atom number out of range in PERTCONSTRAINT03 "
			   " block", "In_Topology", io::message::error);
	}
      
	topology::two_body_term_struct b(i-1, j-1, t_A-1);
	
	std::vector<topology::two_body_term_struct>::iterator b_it
	  = std::find(topo.solute().distance_constraints().begin(), 
		      topo.solute().distance_constraints().end(), 
		      b);
	  
	if (b_it == topo.solute().distance_constraints().end()){
	  io::messages.add("Perturbation of a non-existing distance "
			   "constraint in PERTCONSTRAINT03 block.",
			   "In_Perturbation", io::message::error);
	  
	}
	
	topo.solute().distance_constraints().erase(b_it);
	topology::perturbed_two_body_term_struct 
	  pb(i-1, j-1, t_A-1, t_B-1);

	topo.perturbed_solute().distance_constraints().push_back(pb);

	std::cout << "\t" 
		  << std::setw(10) << pb.i+1 
		  << std::setw(10) << pb.j+1
		  << std::setw(10) << pb.A_type+1 
		  << std::setw(10) << pb.B_type+1 
		  << "\n";
	
      }
    
      if(n != num){
	io::messages.add("Wrong number of bonds in PERTCONSTRAINT03 block",
			 "In_Perturbation", io::message::error);
	throw std::runtime_error("error in PERTCONSTRAINT03 block "
				 "(n != num)");
      }
    }
    
  } // PERTCONSTRAINT03

  { // PERTBANGLE03
    buffer = m_block["PERTBANGLE03"];
    if (buffer.size()){
      std::cout << "\tPERTANGLES\n";

      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;
      
      std::cout << "\t"
		<< std::setw(10) << "atom i"
		<< std::setw(10) << "atom j"
		<< std::setw(10) << "atom k"
		<< std::setw(10) << "type A"
		<< std::setw(10) << "type B"
		<< "\n";

      for(n=0; it != buffer.end() -1; ++it, ++n){
	int i, j, k, t_A, t_B;
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> i >> j >> k >> t_A >> t_B;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTBANGLE03 block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error("bad line in PERTBANGLE03 block");
	}
	
	topology::three_body_term_struct a(i-1, j-1, k-1, t_A-1);
	std::vector<topology::three_body_term_struct>::iterator a_it
	  = std::find(topo.solute().angles().begin(), 
		      topo.solute().angles().end(), 
		      a);
	
	if (a_it == topo.solute().angles().end()){
	  io::messages.add("Perturbation of a non-existing angle in "
			   "PERTBANGLE03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("trying to perturb non-existing angle");
	}
	
	topo.solute().angles().erase(a_it);
	topology::perturbed_three_body_term_struct pa(i-1, j-1, k-1, t_A-1, t_B-1);
	
	std::cout << "\t"
		  << std::setw(10) << pa.i+1 
		  << std::setw(10) << pa.j+1
		  << std::setw(10) << pa.k+1
		  << std::setw(10) << pa.A_type+1 
		  << std::setw(10) << pa.B_type+1 
		  << "\n";
	
	topo.perturbed_solute().angles().push_back(pa);
      }
      if (n != num){
	  io::messages.add("Wrong number of bonds in PERTBANGLE03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in PERTBANGLE03 block (n != num)");
      }
      else if (_lineStream.fail()){
	io::messages.add("Bad line in PERTBANGLE03 block.",
			 "InTopology", io::message::error);
    	throw std::runtime_error("error in PERTBANGLE03 block (fail)");
      }
      
      std::cout << "\tEND\n";
      
    } // if block present
  } // PERTANGLE03
  
  { // PERTIMPDIHEDRAL03
    buffer = m_block["PERTIMPDIHEDRAL03"];
    if (buffer.size()){
      std::cout << "\tPERTIMPDIHEDRALS\n";

      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;
      
      std::cout << "\t"
		<< std::setw(10) << "atom i"
		<< std::setw(10) << "atom j"
		<< std::setw(10) << "atom k"
		<< std::setw(10) << "atom l"
		<< std::setw(10) << "type A"
		<< std::setw(10) << "type B"
		<< "\n";

      for(n=0; it != buffer.end() -1; ++it, ++n){
	int i, j, k, l, t_A, t_B;
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> i >> j >> k >> l >> t_A >> t_B;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTIMPDIHEDRAL03 block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error("bad line in PERTIMPDIHEDRAL03 block");
	}
	
	topology::four_body_term_struct id(i-1, j-1, k-1,l-1, t_A-1);
	std::vector<topology::four_body_term_struct>::iterator id_it
	  = std::find(topo.solute().improper_dihedrals().begin(), 
		      topo.solute().improper_dihedrals().end(), 
		      id);
	
	if (id_it == topo.solute().improper_dihedrals().end()){
	  io::messages.add("Perturbation of a non-existing improper dihedral in "
			   "PERTIMPDIHEDAL03 block.",
			   "InTopology", io::message::error);
	  throw 
	    std::runtime_error("trying to perturb non-existing improper dihedral");
	}
	
	topo.solute().improper_dihedrals().erase(id_it);
	topology::perturbed_four_body_term_struct pid(i-1, j-1, k-1, l-1, 
						      t_A-1, t_B-1);
	
	std::cout << "\t"
		  << std::setw(10) << pid.i+1 
		  << std::setw(10) << pid.j+1
		  << std::setw(10) << pid.k+1
		  << std::setw(10) << pid.l+1
		  << std::setw(10) << pid.A_type+1 
		  << std::setw(10) << pid.B_type+1 
		  << "\n";
	
	topo.perturbed_solute().improper_dihedrals().push_back(pid);
      }
      
      if (n != num){
	  io::messages.add("Wrong number of bonds in PERTIMPDIHEDRAL03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in PERTIMPDIHEDRAL03 block (n != num)");
      }
      else if (_lineStream.fail()){
	io::messages.add("Bad line in PERTIMPDIHEDRAL03 block.",
			 "InTopology", io::message::error);
    	throw std::runtime_error("error in PERTIMPDIHEDRAL03 block (fail)");
      }
      std::cout << "\tEND\n";

    } // if block present
   
  } // PERTIMPDIHEDRAL03

  { // PERTDIHEDRAL03
    buffer = m_block["PERTDIHEDRAL03"];
    if (buffer.size()){
      std::cout << "\tPERTDIHEDRALS\n";

      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;

      std::cout << "\t"
		<< std::setw(10) << "atom i"
		<< std::setw(10) << "atom j"
		<< std::setw(10) << "atom k"
		<< std::setw(10) << "atom l"
		<< std::setw(10) << "type A"
		<< std::setw(10) << "type B"
		<< "\n";
      
      for(n=0; it != buffer.end() -1; ++it, ++n){
	int i, j, k, l, t_A, t_B;
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> i >> j >> k >> l >> t_A >> t_B;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTDIHEDRAL03 block.",
			   "InTopology", io::message::error);	  
	  throw std::runtime_error("bad line in PERTDIHEDRAL03 block");
	}
	
	topology::four_body_term_struct id(i-1, j-1, k-1,l-1, t_A-1);
	std::vector<topology::four_body_term_struct>::iterator id_it
	  = std::find(topo.solute().dihedrals().begin(), 
		      topo.solute().dihedrals().end(), 
		      id);
	
	if (id_it == topo.solute().dihedrals().end()){
	  io::messages.add("Perturbation of a non-existing dihedral in "
			   "PERTDIHEDAL03 block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error(
		     "trying to perturb non-existing dihedral dihedral");
	}
	
	topo.solute().dihedrals().erase(id_it);
	topology::perturbed_four_body_term_struct pid(i-1, j-1, k-1, l-1, 
						      t_A-1, t_B-1);

	std::cout << "\t"
		  << std::setw(10) << pid.i+1 
		  << std::setw(10) << pid.j+1
		  << std::setw(10) << pid.k+1
		  << std::setw(10) << pid.l+1
		  << std::setw(10) << pid.A_type+1 
		  << std::setw(10) << pid.B_type+1 
		  << "\n";
	
	topo.perturbed_solute().dihedrals().push_back(pid);
      }
      
      if (n != num){
	  io::messages.add("Wrong number of bonds in PERTDIHEDRAL03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in PERTDIHEDRAL03 block (n != num)");
      }
      else if (_lineStream.fail()){
	io::messages.add("Bad line in PERTDIHEDRAL03 block.",
			 "InTopology", io::message::error);
    	throw std::runtime_error("error in PERTDIHEDRAL03 block (fail)");
      }
      std::cout << "\tEND\n";

    } // if block present
   
  } // PERTDIHEDRAL03

  { // PERTATOMPAIR03
    // has to be read in before(!!) PERTATOM03
    // because the exclusions and 1,4 exclusions have to be adapted...

    buffer = m_block["PERTATOMPAIR03"];
    if (buffer.size()){
      std::cout << "\tPERTATOMPAIRS\n";

      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;

      int i, j, A, B;

      std::cout << "\t"
		<< std::setw(10) << "atom i"
		<< std::setw(10) << "atom j"
		<< std::setw(10) << "type A"
		<< std::setw(10) << "type B"
		<< "\n";

      for(n = 0; it != buffer.end() - 1; ++it, ++n){
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> i >> j >> A >> B;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTATOMPAIR03 block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error("bad line in PERTATOMPAIR03 block\n\t"+ *it);
	}
	
	if(j<i) { int t=j; j=i; i=t; }

	topology::perturbed_two_body_term_struct ap(i-1,j-1,A,B);
	
	std::cout << "\t"
		  << std::setw(10) << ap.i+1
		  << std::setw(10) << ap.j+1
		  << std::setw(10) << ap.A_type
		  << std::setw(10) << ap.B_type
		  << std::endl;
	
	topo.perturbed_solute().atompairs().push_back(ap);

	// make sure it's excluded
	if (topo.all_exclusion(ap.i).count(ap.j) != 1){
	  topo.all_exclusion(ap.i).insert(ap.j);
	  DEBUG(7, "excluding perturbed pair " << ap.i << " and " << ap.j);
	  
	}
	else{
	  // it was already excluded, let's remove it from the
	  // exclusions or 1,4 pairs...
	  
	  // is it in the exclusions
	  if (topo.exclusion(ap.i).count(ap.j)){
	    DEBUG(7, "removing perturbed pair from exclusion " 
		  << ap.i << " and " << ap.j);
	    topo.exclusion(ap.i).erase(ap.j);
	  }
	  if (topo.one_four_pair(ap.i).count(ap.j)){
	    DEBUG(7, "removing perturbed pair from one four " 
		  << ap.i << " and " << ap.j);
	    topo.one_four_pair(ap.i).erase(ap.j);
	  }
	  
	}
      }
      
      if (n != num){
	  io::messages.add("Wrong number of bonds in PERTATOMPAIR03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in PERTATOMPAIR03 block (n != num)");
      }
      else if (_lineStream.fail()){
	io::messages.add("Bad line in PERTATOMPAIR03 block.",
			 "InTopology", io::message::error);
    	throw std::runtime_error("error in PERTATOMPAIR03 block (fail)");
      }
      std::cout << "\tEND\n";
      
    } // if block present
  } // PERTATOMPAIR03
  
  { // PERTATOM03
    
    buffer = m_block["PERTATOM03"];
    if (buffer.size()){
      std::cout << "\tPERTATOMS\n";
      DEBUG(7, "PERTATOM03 block");
      
      
      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;
      
      size_t seq, res, a_iac, b_iac;
      double a_mass, b_mass, a_charge, b_charge;
      double lj_soft, crf_soft;
      std::string name;

      std::cout << "\t"
		<< std::setw(5) << "seq"
		<< std::setw(8) << "IAC(A)"
		<< std::setw(10) << "mass(A)"
		<< std::setw(10) << "charge(A)"
		<< std::setw(8) << "IAC(B)"
		<< std::setw(10) << "mass(B)"
		<< std::setw(10) << "charge(B)"
		<< std::setw(10) << "LJ(soft)"
		<< std::setw(10) << "CRF(soft)"
		<< "\n";
      
      for(n = 0; it != buffer.end() - 1; ++it, ++n){
	DEBUG(10, "\treading a line: " << n);
	
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> seq >> res >> name >> a_iac >> a_mass >> a_charge
		    >> b_iac >> b_mass >> b_charge
		    >> lj_soft >> crf_soft;
	
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in PERTATOM03 block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error("bad line in PERTATOM03 block\n"+*it);
	}
	
	--seq;
	--a_iac;
	--b_iac;
	topology::Perturbed_Atom atom(seq, a_iac, a_mass, a_charge,
					b_iac, b_mass, b_charge,
					lj_soft, crf_soft);

	DEBUG(10, "\tcreated an atom");
	
	std::cout << "\t"
		  << std::setw(5) << seq + 1
		  << std::setw(8) << a_iac + 1
		  << std::setw(10) << a_mass
		  << std::setw(10) << a_charge
		  << std::setw(8) << b_iac + 1
		  << std::setw(10) << b_mass
		  << std::setw(10) << b_charge
		  << std::setw(10) << lj_soft
		  << std::setw(10) << crf_soft
		  << "\n";
	
	atom.exclusion() = topo.exclusion(seq);
	topo.exclusion(seq).clear();
	DEBUG(10, "\treplace the exclusions to perturbation");

	std::vector<std::set<int> > & ex = topo.exclusion();
	int seq2=0;
	
	for(std::vector<std::set<int> >::iterator eit=ex.begin(),
	      eto=ex.end(); eit!=eto; ++eit, ++seq2){
	  if(eit->count(seq)){
	    atom.exclusion().insert(seq2);
	    eit->erase(seq);
	  }
	}
	DEBUG(10, "\tadapted perturbed exclusions");
	
	atom.one_four_pair() = topo.one_four_pair(seq);
	topo.one_four_pair(seq).clear();
	DEBUG(10, "\treplaced the 14 interactions");
	
	std::vector<std::set<int> > & ofp = topo.one_four_pair();
	seq2=0;
	
	for(std::vector<std::set<int> >::iterator pit=ofp.begin(), 
	      pito= ofp.end(); pit!=pito; ++pit, ++seq2){
	  if(pit->count(seq)){
	    atom.one_four_pair().insert(seq2);
	    pit->erase(seq);
	  }
	}
	DEBUG(10, "\tadapted 14 interactions");
	
	
	topo.perturbed_solute().atoms()[seq] = atom;

	assert(seq<topo.is_perturbed().size());
	topo.is_perturbed()[seq] = true;
	
      }
      if (n != num){
	  io::messages.add("Wrong number of bonds in PERTATOM03 block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in PERTATOM03 block (n != num)");
      }
      else if (_lineStream.fail()){
	io::messages.add("Bad line in PERTATOM03 block.",
			 "InTopology", io::message::error);
    	throw std::runtime_error("error in PERTATOM03 block (fail)");
      }

      std::cout << "\tEND\n";
      
    } // if block present
    
  } // PERTATOM03
    
  { // SCALEDINTERACTIONS

    buffer = m_block["SCALEDINTERACTIONS"];
    if (buffer.size()){
      if(!param.perturbation.scaling){
	io::messages.add("Scaled interactions not turned on, ignoring SCALEDINTERACTIONS block.",
			 "InPerturbation", io::message::warning);
      }
      else{
	std::cout << "\tSCALED INTERACTIONS\n";
	
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	int num, n;
	_lineStream >> num;
	++it;
	
	int i, j;
	double A, B;
	
	std::cout << "\t"
		  << std::setw(10) << "group i"
		  << std::setw(10) << "group j"
		  << std::setw(10) << "scale A"
		  << std::setw(10) << "scale B"
		  << "\n";
	
	for(n = 0; it != buffer.end() - 1; ++it, ++n){
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> A >> B;
	  
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in PERTATOM03 block.",
			     "InTopology", io::message::error);
	    throw std::runtime_error("bad line in SCALEDINTERACTIONS block\n\t"
				     + *it);
	  }
	  
	  --i;
	  --j;
	  
	  std::pair<int, int> energy_pair(i,j);
	  std::pair<int, int> energy_pair2(j,i);
	  
	  std::pair<double, double> scale_pair(A,B);
	  
	  topo.energy_group_scaling()[energy_pair]=scale_pair;
	  topo.energy_group_scaling()[energy_pair2]=scale_pair;
	  
	  std::cout << "\t"
		    << std::setw(10) << i+1
		    << std::setw(10) << j+1
		    << std::setw(10) << A
		    << std::setw(10) << B
		    << std::endl;
	  
	}
	
	if (n != num){
	  io::messages.add("Wrong number of bonds in SCALEDINTERACTIONS block.",
			   "InTopology", io::message::error);	
	  throw std::runtime_error("error in SCALEDINTERACTIONS block (n != num)");
	}
	else if (_lineStream.fail()){
	  io::messages.add("Bad line in SCALEDINTERACTIONS block.",
			   "InTopology", io::message::error);
	  throw std::runtime_error("error in SCALEDINTERACTIONS block (fail)");
	}
	std::cout << "\tEND\n";
      } // if scaling turned on
      
    } // if block present
    else{
      if(param.perturbation.scaling){
	io::messages.add("Scaling turned on but no SCALEDINTERACTIONS block.",
			 "InPerturbation", io::message::error);
      }
    }
  } // PERTATOMPAIR03

  std::cout << "END\n";

  // and update the properties for lambda
  topo.update_for_lambda();
  
}


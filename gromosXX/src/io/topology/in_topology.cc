/**
 * @file in_topology.cc
 * implements methods of In_Topology.
 */


#include <stdheader.h>

#include <topology/topology.h>
#include <simulation/multibath.h>
#include <simulation/parameter.h>
#include <interaction/interaction_types.h>
#include <io/instream.h>
#include <util/parse_tcouple.h>

#include <io/blockinput.h>

#include "in_topology.h"

#undef MODULE
#undef SUBMODULE
#define MODULE io
#define SUBMODULE topology


template<typename T>
bool check_type(std::vector<std::string> const & buffer, std::vector<T> term)
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
      
      if (int(it->type) >= num){
	return false;
      }
    }
  }
  else return false;
  return true;
}

void 
io::In_Topology::read(topology::Topology& topo,
		      simulation::Parameter &param){

  DEBUG(7, "reading in topology");

  if (!quiet){
    std::cout << "TOPOLOGY\n";
    std::cout << title << "\n";
  }
  
  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  // double start = util::now();
  
  { // TOPPHYSCON
    buffer = m_block["TOPPHYSCON"];
    
    std::string s;
    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1,
				buffer.end()-1, s));
    double four_pi_eps0_i;
    
    _lineStream >> four_pi_eps0_i >> math::h_bar;
    math::four_pi_eps_i = four_pi_eps0_i / param.longrange.epsilon;
    
    if (_lineStream.fail())
      io::messages.add("Bad line in TOPPHYSCON block",
		       "InTopology", io::message::error);
  }

  if (param.system.npm){
    
    { // RESNAME
      if (!quiet)
	std::cout << "\tRESNAME\n\t";
    
      DEBUG(10, "RESNAME block");
      buffer = m_block["RESNAME"];
      it = buffer.begin()+1;
      int n, num;
      _lineStream.clear();
      _lineStream.str(*it);
      _lineStream >> num;
      ++it;
    
      if (!quiet && num > 10){
	for(n=0; n<10; ++n)
	  std::cout << std::setw(8) << n+1;
	std::cout << "\n\t";
      }
    
      for(n=0; it != buffer.end() - 1; ++it, ++n){
	std::string s;
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> s;
      
	if (!quiet){
	  if (n && ((n) % 10) == 0) std::cout << std::setw(10) << n << "\n\t";
	  std::cout << std::setw(8) << s;      
	}
      
	topo.residue_names().push_back(s);
      }

      if (n != num){
	io::messages.add("Error in RESNAME block: n!=num.",
			 "InTopology", io::message::error);
      }

      if (!quiet)
	std::cout << "\n\tEND\n";
    
    
    } // RESNAME

    // std::cout << "time after RESNAME: " << util::now() - start << std::endl;

    { // SOLUTEATOM
      DEBUG(10, "SOLUTEATOM block");
      buffer = m_block["SOLUTEATOM"];
  
      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      topo.resize(num);

      if (!quiet)
	std::cout << "\tSOLUTEATOM\n\t"
		  << "\tnumber of atoms : " << num;
    
      // put the rest of the block into a single stream
      ++it;

      // std::string soluteAtoms;
      // concatenate(it, buffer.end()-1, soluteAtoms);

      // _lineStream.clear();
      // _lineStream.str(soluteAtoms);

      // std::cout << "\ntime after concatenate: " << util::now() - start << std::endl;

      int a_nr, r_nr, t, cg, n_ex, a_ex;
      double m, q;
      std::string s;
      std::set<int> ex;
      std::set<int> ex14;
      
      for(n=0; n < num; ++n){

	_lineStream.clear();
	_lineStream.str(*it);
	++it;
      
	_lineStream >> a_nr >> r_nr >> s >> t >> m >> q >> cg;

	if (a_nr != n+1){
	  io::messages.add("Error in SOLUTEATOM block: atom number not sequential.",
			   "InTopology", io::message::error);
	}

	if (r_nr > int(topo.residue_names().size()) || r_nr < 1){
	  io::messages.add("Error in SOLUTEATOM block: residue number out of range.",
			   "InTopology", io::message::error);
	}
      
	if (t < 1){
	  io::messages.add("Error in SOLUTEATOM block: iac < 1.",
			   "InTopology", io::message::error);
	}

	if (m <= 0){
	  io::messages.add("Error in SOLUTEATOM block: mass < 0.",
			   "InTopology", io::message::error);
	}

	if (cg != 0 && cg != 1){
	  io::messages.add("Error in SOLUTEATOM block: cg = 0 / 1.",
			   "InTopology", io::message::error);
	}

	if (!(_lineStream >> n_ex)){
	  if (_lineStream.eof()){
	    _lineStream.clear();
	    _lineStream.str(*it);
	    ++it;
	    _lineStream >> n_ex;
	  }
	  else{
	    io::messages.add("Error in SOLUTEATOM block: number of exclusions "
			     "could not be read.",
			     "InTopology", io::message::error);	
	  }
	}

	if (n_ex < 0){
	  io::messages.add("Error in SOLUTEATOM block: number of exclusions < 0.",
			   "InTopology", io::message::error);
	}
      
	// exclusions
	ex.clear();
	for(int i=0; i<n_ex; ++i){
	  if (!(_lineStream >> a_ex)){
	    if (_lineStream.eof()){
	      _lineStream.clear();
	      _lineStream.str(*it);
	      ++it;
	      _lineStream >> a_ex;
	    }
	    else{
	      io::messages.add("Error in SOLUTEATOM block: exclusion "
			       "could not be read.",
			       "InTopology", io::message::error);	
	    }
	  }

	  if (a_ex <= a_nr)
	    io::messages.add("Error in SOLUTEATOM block: exclusions only to "
			     "larger atom numbers.",
			     "InTopology", io::message::error);	

	  ex.insert(a_ex-1);
	}
      
	// 1,4 - pairs
	if (!(_lineStream >> n_ex)){
	  if (_lineStream.eof()){
	    _lineStream.clear();
	    _lineStream.str(*it);
	    ++it;
	    _lineStream >> n_ex;
	  }
	  else{
	    io::messages.add("Error in SOLUTEATOM block: number of 1,4 - exclusion "
			     "could not be read.",
			     "InTopology", io::message::error);	
	  }
	}

	if (n_ex < 0){
	  io::messages.add("Error in SOLUTEATOM block: number of 1,4 exclusions < 0.",
			   "InTopology", io::message::error);
	}

	ex14.clear();
	for(int i=0; i<n_ex; ++i){
	  if (!(_lineStream >> a_ex)){
	    if (_lineStream.eof()){
	      _lineStream.clear();
	      _lineStream.str(*it);
	      ++it;
	      _lineStream >> a_ex;
	    }
	    else{
	      io::messages.add("Error in SOLUTEATOM block: 1,4 - exclusion "
			       "could not be read.",
			       "InTopology", io::message::error);	
	    }
	  }

	  if (a_ex <= a_nr)
	    io::messages.add("Error in SOLUTEATOM block: 1,4 - exclusions only to "
			     "larger atom numbers.",
			     "InTopology", io::message::error);	
	
	  ex14.insert(a_ex-1);
	}
      
	if (_lineStream.fail())
	  io::messages.add("bad line in SOLUTEATOM block",
			   "In_Topology",
			   io::message::critical);

	topo.add_solute_atom(s, r_nr-1, t-1, m, q, cg, ex, ex14);
      }
      if (!quiet)
	std::cout << "\n\tEND\n";
    
    } // SOLUTEATOM
  
    // std::cout << "time after SOLUTEATOM: " << util::now() - start << std::endl;
  
    { // BONDH
      DEBUG(10, "BONDH block");

      if (!quiet)
	std::cout << "\tBOND";
    
      buffer.clear();
      buffer = m_block["BONDH"];
      if (buffer.size()){
	it = buffer.begin() + 1;
      
	_lineStream.clear();
	_lineStream.str(*it);
      
	int num, n;
	_lineStream >> num;
	++it;
      
	if (!quiet){
	  if (param.constraint.ntc == 2 || param.constraint.ntc == 3){
	    std::cout << "\n\t\t"
		      << num
		      << " bonds from BONDH block added to CONSTRAINT";
	  }
	  else
	    std::cout << "\n\t\tbonds containing hydrogens : "
		      << num;
	}
      
	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, t;
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> t;
	
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in BONDH block",
			     "In_Topology", io::message::error);
	  }
	
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1){
	    io::messages.add("Atom number out of range in BONDH block",
			     "In_Topology", io::message::error);
	  }
	
	  if (param.constraint.ntc == 2 || param.constraint.ntc == 3){
	    topo.solute().distance_constraints().
	      push_back(topology::two_body_term_struct(i-1, j-1, t-1));
	  }
	  else
	    topo.solute().bonds().
	      push_back(topology::two_body_term_struct(i-1, j-1, t-1));
	
	}
      
	if(n != num){
	  io::messages.add("Wrong number of bonds in BONDH block",
			   "In_Topology", io::message::error);
	}
      }

    } // BONDH
  
    { // BOND
      DEBUG(10, "BOND block");
      buffer = m_block["BOND"];

      if (buffer.size()){
      
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet){
	  if (param.constraint.ntc == 3){
	    std::cout << "\n\t\t"
		      << num
		      << " bonds from BOND block added to CONSTRAINT";
	  }
	  else
	    std::cout << "\n\t\tbonds not containing hydrogens : "
		      << num;
	}
      
	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, t;
	
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> t;
	
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in BOND block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1){
	    io::messages.add("Atom number out of range in BOND block",
			     "In_Topology", io::message::error);
	  }
      
	  if (param.constraint.ntc == 3){
	    topo.solute().distance_constraints().
	      push_back(topology::two_body_term_struct(i-1, j-1, t-1));
	  }
	  else
	    topo.solute().bonds().
	      push_back(topology::two_body_term_struct(i-1, j-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in BOND block",
			   "In_Topology", io::message::error);
	}
      }
  
      if (!quiet)
	std::cout << "\n\tEND\n";

    } // BOND

    // std::cout << "time after BONDS: " << util::now() - start << std::endl;

    // check the bonds
    if (!check_type(m_block["BONDTYPE"], topo.solute().bonds())){
      io::messages.add("Illegal bond type in BOND(H) block",
		       "In_Topology", io::message::error);
    }

    // std::cout << "time after CHECKBONDS: " << util::now() - start << std::endl;

    { // CONSTRAINT
      DEBUG(10, "CONSTRAINT block");
      buffer = m_block["CONSTRAINT"];
  
      if (buffer.size() && param.constraint.ntc != 1){
      
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet)
	  std::cout << "\tCONSTRAINT\n\t\t"
		    << num
		    << " bonds in CONSTRAINT block."
		    << "\n\t\ttotal of constraint bonds : " 
		    << num + unsigned(topo.solute().distance_constraints().size())
		    << "\n\tEND\n";
      
	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, t;
	
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> t;
	
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in CONSTRAINT block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1){
	    io::messages.add("Atom number out of range in CONSTRAINT block",
			     "In_Topology", io::message::error);
	  }
      
	  topo.solute().distance_constraints().
	    push_back(topology::two_body_term_struct(i-1, j-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in CONSTRAINT block",
			   "In_Topology", io::message::error);
	}
      }
    
    } // CONSTRAINT

    // std::cout << "time after CONSTRAINTS: " << util::now() - start << std::endl;

    // check the bonds in constraints
    if (!check_type(m_block["BONDTYPE"], topo.solute().distance_constraints())){
      io::messages.add("Illegal bond type in CONSTRAINT (or BOND(H)) block",
		       "In_Topology", io::message::error);
    }

    // std::cout << "time after check CONSTRAINTS: " << util::now() - start << std::endl;

    { // BONDANGLEH

      if (!quiet)
	std::cout << "\tBONDANGLE";

      DEBUG(10, "BONDANGLEH block");
      buffer.clear();
      buffer = m_block["BONDANGLEH"];
  
      if(buffer.size()){
      
	it = buffer.begin() + 1;

	_lineStream.clear();
	_lineStream.str(*it);

	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet)
	  std::cout << "\n\t\tbondangles not containing hydrogens : " << num;

	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, k, t;
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> k >> t;
      
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in BONDANGLEH block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      k > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1 || k < 1){
	    io::messages.add("Atom number out of range in BONDANGLEH block",
			     "In_Topology", io::message::error);
	  }

	  topo.solute().angles().
	    push_back(topology::three_body_term_struct(i-1, j-1, k-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in BONDANGLEH block",
			   "In_Topology", io::message::error);
	}
      }
        
    } // BONDANGLEH
  
    { // BONDANGLE
      DEBUG(10, "BONDANGLE block");
      buffer = m_block["BONDANGLE"];
  
      if (buffer.size()){
      
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet)
	  std::cout << "\n\t\tbondangles containing hydrogens : " << num;
    
	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, k, t;
      
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> k >> t;
      
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in BONDANGLE block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      k > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1 || k < 1){
	    io::messages.add("Atom number out of range in BONDANGLE block",
			     "In_Topology", io::message::error);
	  }
      
	  topo.solute().angles().
	    push_back(topology::three_body_term_struct(i-1, j-1, k-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in BONDANGLE block",
			   "In_Topology", io::message::error);
	}
      }

      if (!quiet)
	std::cout << "\n\tEND\n";

    } // BONDANGLE

    // std::cout << "time after BONDANGLE: " << util::now() - start << std::endl;

    // check the angles
    if (!check_type(m_block["BONDANGLETYPE"], topo.solute().angles())){
      io::messages.add("Illegal bond angle type in BONDANGLE(H) block",
		       "In_Topology", io::message::error);
    }

    // std::cout << "time after check BONDANGLE: " << util::now() - start << std::endl;

    { // IMPDIHEDRAL
      DEBUG(10, "IMPDIHEDRAL block");
      buffer = m_block["IMPDIHEDRAL"];
  
      if (!quiet)
	std::cout << "\tIMPDIHEDRAL";

      if(buffer.size()){
      
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet)
	  std::cout << "\n\t\timproper dihedrals not containing hydrogens : "
		    << num;
    
	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, k, l, t;
      
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> k >> l >> t;
      
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in IMPDIHEDRAL block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      k > int(topo.num_solute_atoms()) || l > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1 || k < 1 || l < 1){
	    io::messages.add("Atom number out of range in IMPDIHEDRAL block",
			     "In_Topology", io::message::error);
	  }
      
	  topo.solute().improper_dihedrals().
	    push_back(topology::four_body_term_struct(i-1, j-1, k-1, l-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in IMPDIHEDRAL block",
			   "In_Topology", io::message::error);
	}
      }
    
    } // IMPDIHEDRAL

    { // IMPDIHEDRALH
      DEBUG(10, "IMPDIHEDRALH block");
      buffer.clear();
      buffer = m_block["IMPDIHEDRALH"];
  
      if(buffer.size()){
      
	it = buffer.begin() + 1;

	_lineStream.clear();
	_lineStream.str(*it);

	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet)
	  std::cout << "\n\t\timproper dihedrals containing hydrogens : "
		    << num;

	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, k, l, t;
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> k >> l >> t;
      
      
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in IMPDIHEDRALH block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      k > int(topo.num_solute_atoms()) || l > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1 || k < 1 || l < 1){
	    io::messages.add("Atom number out of range in IMPDIHEDRALH block",
			     "In_Topology", io::message::error);
	  }
      
	  topo.solute().improper_dihedrals().
	    push_back(topology::four_body_term_struct(i-1, j-1, k-1, l-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in IMPDIHEDRALH block",
			   "In_Topology", io::message::error);
	}
      }

      if (!quiet)
	std::cout << "\n\tEND\n";
    
    } // IMPDIHEDRALH

    // check the imporopers
    if (!check_type(m_block["IMPDIHEDRALTYPE"], topo.solute().improper_dihedrals())){
      io::messages.add("Illegal improper dihedral type in IMPDIHEDRAL(H) block",
		       "In_Topology", io::message::error);
    }  

    { // DIHEDRAL
      DEBUG(10, "DIHEDRAL block");    
      buffer = m_block["DIHEDRAL"];

      if (!quiet)
	std::cout << "\tDIHEDRAL";
    
      if(buffer.size()){
      
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	int num, n;
	_lineStream >> num;
	++it;
    
	if (!quiet)
	  std::cout << "\n\t\tdihedrals not containing hydrogens : "
		    << num;

	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, k, l, t;
      
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> k >> l >> t;
      
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in DIHEDRAL block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      k > int(topo.num_solute_atoms()) || l > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1 || k < 1 || l < 1){
	    io::messages.add("Atom number out of range in DIHEDRAL block",
			     "In_Topology", io::message::error);
	  }
      
	  topo.solute().dihedrals().
	    push_back(topology::four_body_term_struct(i-1, j-1, k-1, l-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in DIHEDRAL block",
			   "In_Topology", io::message::error);
	}
      }
    
    } // DIHEDRAL

    { // DIHEDRALH
      DEBUG(10, "DIHEDRALH block");
      buffer.clear();
      buffer = m_block["DIHEDRALH"];
      if(buffer.size()){
      
	it = buffer.begin() + 1;

	_lineStream.clear();
	_lineStream.str(*it);

	int num, n;
	_lineStream >> num;
	++it;

	if (!quiet)
	  std::cout << "\n\t\tdihedrals containing hydrogens : "
		    << num;

	for(n=0; it != buffer.end() - 1; ++it, ++n){
	  int i, j, k, l, t;
	  _lineStream.clear();
	  _lineStream.str(*it);
	  _lineStream >> i >> j >> k >> l >> t;
      
	  if (_lineStream.fail() || ! _lineStream.eof()){
	    io::messages.add("Bad line in DIHEDRALH block",
			     "In_Topology", io::message::error);
	  }
      
	  if (i > int(topo.num_solute_atoms()) || j > int(topo.num_solute_atoms()) ||
	      k > int(topo.num_solute_atoms()) || l > int(topo.num_solute_atoms()) ||
	      i < 1 || j < 1 || k < 1 || l < 1){
	    io::messages.add("Atom number out of range in DIHEDRALH block",
			     "In_Topology", io::message::error);
	  }
      
	  topo.solute().dihedrals().
	    push_back(topology::four_body_term_struct(i-1, j-1, k-1, l-1, t-1));
	}
    
	if(n != num){
	  io::messages.add("Wrong number of bonds in DIHEDRALH block",
			   "In_Topology", io::message::error);
	}
      }
    
      if (!quiet)
	std::cout << "\n\tEND\n";
    
    } // DIHEDRALH

    // std::cout << "time after DIHEDRAL: " << util::now() - start << std::endl;

    // check the dihedrals
    if (!check_type(m_block["DIHEDRALTYPE"], topo.solute().dihedrals())){
      io::messages.add("Illegal dihedral type in DIHEDRAL(H) block",
		       "In_Topology", io::message::error);
    }

    // add the submolecules (should be done before solvate ;-)
    topo.molecules() = param.submolecules.submolecules;

    if (topo.molecules().size() == 0){
      topo.molecules().push_back(0);
      topo.molecules().push_back(topo.num_solute_atoms());
    }

    // submolecules check
    if (topo.molecules().back()
	!= topo.num_solute_atoms()){
    
      io::messages.add("Error in SUBMOLECULE block: "
		       "last submolecule has to end with last solute atom",
		       "In_Topology", io::message::error);
    }

  
  } // npm != 0
    
  { // SOLVENTATOM and SOLVENTCONSTR
    // give it a number (SOLVENTATOM1, SOLVENTATOM2) for multiple
    // solvents...
    DEBUG(10, "SOLVENTATOM block");
    buffer = m_block["SOLVENTATOM"];

    if (!quiet)
      std::cout << "\tSOLVENT";
    
    if (buffer.size()){
      
      unsigned int res_nr = unsigned(topo.residue_names().size());
    
      topo.residue_names().push_back("SOLV");

      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);
      int num, n;
      _lineStream >> num;
      ++it;
    
      if (!quiet)
	std::cout << "\n\t\tatoms : " << num;

      topology::Solvent s;
    
      std::string name;
      int i, iac;
      double mass, charge;
    
      for(n=0; it != buffer.end()-1; ++it, ++n){
	_lineStream.clear();
	_lineStream.str(*it);
      
	_lineStream >> i >> name >> iac >> mass >> charge;

	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in SOLVENTATOM block",
			   "In_Topology", io::message::error);	
	}
      

	s.add_atom(name, res_nr, iac-1, mass, charge);
      }
    
      if (n!=num){
	io::messages.add("Error in SOLVENTATOM block (num != n)",
			 "In_Topology", io::message::error);
      }
    
      // lookup the number of bond types
      // add additional ones for the solvent constraints
      int num_bond_types;
    
      buffer = m_block["BONDTYPE"];
      if (!buffer.size())
	num_bond_types = 0;
      else{
	it = buffer.begin() + 1;
	_lineStream.clear();
	_lineStream.str(*it);
	_lineStream >> num_bond_types;
      
	if (num_bond_types < 0){
	  io::messages.add("Illegal value for number of bond types in BONDTYPE block",
			   "In_Topology", io::message::error);	
	  num_bond_types = 0;
	}
      }
    
      buffer = m_block["SOLVENTCONSTR"];
      if (!buffer.size()){
	io::messages.add("SOLVENTCONSTR block missing.",
			 "In_Topology", io::message::error);	

      }
    
      it = buffer.begin() + 1;
      _lineStream.clear();
      _lineStream.str(*it);

      _lineStream >> num;
      ++it;
    
      if (!quiet)
	std::cout << "\n\t\tconstraints : " << num;
      
      int j;
      double b0;
    
      for(n=0; it != buffer.end()-1; ++it, ++n){
	_lineStream.clear();
	_lineStream.str(*it);
      
	_lineStream >> i >> j >> b0;
      
	if (_lineStream.fail() || ! _lineStream.eof()){
	  io::messages.add("Bad line in SOLVENTCONSTR block",
			   "In_Topology", io::message::error);	
	}
      
	// the solvent (distance constraints) bond types
	s.add_distance_constraint
	  (topology::two_body_term_struct(i-1, j-1, num_bond_types + n));
      }

      if (n!=num){
	io::messages.add("Error in SOLVENTCONSTR block (num != n)",
			 "In_Topology", io::message::error);
      }
      topo.add_solvent(s);
    }

  }

  // add the solvent to the topology
  if (!quiet)
    std::cout << "\n\t\tadding " << param.system.nsm 
	      << " solvents.";
  
  // if (param.system.nsm) 
  topo.solvate(0, param.system.nsm);  
  
  if (!quiet)
    std::cout << "\n\tEND\n";

  // set lambda (new and old one, yes it looks strange...)
  topo.lambda(param.perturbation.lambda);
  topo.lambda(param.perturbation.lambda);
  
  topo.lambda_exp(param.perturbation.lambda_exponent);

  //==================================================
  // CHECKING
  //==================================================
    
  // submolecules check
  if (topo.molecules().back()
      != topo.num_atoms()){
    
    io::messages.add("Error in SUBMOLECULE / solvation block: "
		     "last submolecule has to end with last atom",
		     "In_Topology", io::message::error);
  }

  // chargegroup check (starts with 0)
  if (topo.chargegroups()[topo.num_solute_chargegroups()] != int(topo.num_solute_atoms())){
    io::messages.add("Error: last solute atom has to be end of chargegroup",
		     "In_Topology",
		     io::message::error);
    std::cout << "ERROR:"
	      << "\tsolute cg    : " << topo.num_solute_chargegroups() << "\n"
	      << "\tsolute atoms : " << topo.num_solute_atoms() << "\n"
	      << "\tlast cg      : " << topo.chargegroups()[topo.num_solute_chargegroups()] << "\n";
  }

  if (!quiet)
    std::cout << "\n\tSOLUTE [sub]molecules: " 
	      << unsigned(topo.molecules().size()) - param.system.nsm - 1 << "\n";

  DEBUG(10, "molecules().size: " << unsigned(topo.molecules().size())
	<< " nsm : " << param.system.nsm);

  // energy group check
  if (param.force.energy_group.size() == 0){
    param.force.energy_group.push_back(topo.num_atoms() -1);
  }

  if (param.force.energy_group.back() != topo.num_atoms()-1){
    io::messages.add("Error in FORCE block: "
		     "last energy group has to end with last atom",
		     "In_Topology", io::message::error);
  }
  // and add them
  unsigned int atom = 0;
  for(unsigned int i=0; i<param.force.energy_group.size(); ++i){
    topo.energy_groups().push_back(param.force.energy_group[i]);
    for( ; atom <= param.force.energy_group[i]; ++atom){
      topo.atom_energy_group().push_back(i);
      // DEBUG(11, "atom " << atom << ": " << i);
    }
  }

  if(!param.multibath.found_multibath && param.multibath.found_tcouple){
    if (!quiet)
      std::cout << "\tparsing a (deprecated) TCOUPLE block into the new "
		<< "MULTIBATH format.\n";
    
    util::parse_TCOUPLE(param, topo);
  }

  if (!quiet)
    std::cout << "END\n";
  
}

void io::In_Topology
::read_harmonic_bonds(std::vector<interaction::bond_type_struct> &b)
{
  
  DEBUG(10, "(HARM)BONDTYPE block");

  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  buffer = m_block["HARMBONDTYPE"];
  if (buffer.size()){
    DEBUG(7, "reading in a DIRK (HARMBONDTYPE) block)");
    io::messages.add("harmonic bond force constants from HARMBONDTYPE block",
		     "In_Topology::bondtype", io::message::notice);
    
    int num, n=0;
    it = buffer.begin()+1;
    _lineStream.clear();
    _lineStream.str(*it);
    _lineStream >> num;
    ++it;
    for(; it!=buffer.end()-1; ++it, ++n){
      double k, r;
      _lineStream.clear();
      _lineStream.str(*it);
      _lineStream >> k >> r;

      if (_lineStream.fail() || ! _lineStream.eof()){
	io::messages.add("bad line in HARMBONDTYPE block",
			 "In_Topology",
			 io::message::error);
	k = 0;
	r = 0;
      }

      // and add...
      b.push_back(interaction::bond_type_struct(k, r));
    }
  
    if (num != n)
      io::messages.add("not enough bond types in HARMBONDTYPE block",
		       "In_Topology",
		       io::message::error);
  }
  else{
    buffer = m_block["BONDTYPE"];

    /*
      io::messages.add("converting bond force constants from quartic "
      "to harmonic form", "InTopology::bondtype",
      io::message::notice);
    */

    if (buffer.size()==0)
      io::messages.add("BONDTYPE block not found!",
		       "In_Topology",
		       io::message::error);

    // 1. BONDTYPE 2. number of types
    for (it = buffer.begin() + 2; 
	 it != buffer.end() - 1; ++it) {

      double k, r;
      _lineStream.clear();
      _lineStream.str(*it);
      
      _lineStream >> k >> r;
      
      if (_lineStream.fail()){
	std::cout << *it << std::endl;
	io::messages.add("bad line in BONDTYPE block!",
			 "In_Topology",
			 io::message::error);
      }
      if (! _lineStream.eof()){
	std::cout << *it << std::endl;
	io::messages.add("eof not reached in BONDTYPE block",
			 "InTopology", io::message::warning);
      }

      // we are reading into harmonic bond term, so convert k
      k *= 2 * r * r;
      
      // and add...
      b.push_back(interaction::bond_type_struct(k, r));
    }
  }

  // also add the solent constraints to the bond types...
  // (if there is one)
  buffer = m_block["SOLVENTCONSTR"];
  if (buffer.size()){
    
    it = buffer.begin() + 1;
    _lineStream.clear();
    _lineStream.str(*it);

    int num;
    _lineStream >> num;
    ++it;
    
    int i, j, n;
    double b0;
    
    for(n=0; it != buffer.end()-1; ++it, ++n){
      _lineStream.clear();
      _lineStream.str(*it);
      
      _lineStream >> i >> j >> b0;
      
      if (_lineStream.fail() || ! _lineStream.eof()){
	io::messages.add("Bad line in SOLVENTCONSTR block",
			 "In_Topology", io::message::error);	
      }
      
      // the solvent (distance constraints) bond types
      b.push_back(interaction::bond_type_struct(0, b0));
      // (K is set to 0.0)
    }

    if (n!=num){
      io::messages.add("Error in SOLVENTCONSTR block (num != n)",
		       "In_Topology", io::message::error);
      
    }
  }
  
}

void io::In_Topology
::read_g96_bonds(std::vector<interaction::bond_type_struct> &b)
{
  
  DEBUG(10, "BONDTYPE block");

  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  buffer = m_block["BONDTYPE"];

  if (buffer.size()==0)
    io::messages.add("BONDTYPE block not found!",
		     "In_Topology",
		     io::message::error);

  // 1. BONDTYPE 2. number of types
  for (it = buffer.begin() + 2; 
       it != buffer.end() - 1; ++it) {

    double k, r;
    _lineStream.clear();
    _lineStream.str(*it);
      
    _lineStream >> k >> r;
      
    if (_lineStream.fail()){
      std::cout << *it << std::endl;
      io::messages.add("bad line in BONDTYPE block",
		       "In_Topology",
		       io::message::error);
    }
    if (! _lineStream.eof()){
      std::cout << *it << std::endl;
      io::messages.add("eof not reached in BONDTYPE block",
		       "InTopology", io::message::warning);
    }

    // and add...
    b.push_back(interaction::bond_type_struct(k, r));
  }

  // also add the solent constraints to the bond types...
  // (if there is one)
  buffer = m_block["SOLVENTCONSTR"];
  if (buffer.size()){
    
    it = buffer.begin() + 1;
    _lineStream.clear();
    _lineStream.str(*it);
    
    int num;
    _lineStream >> num;
    ++it;
    
    int i,j, n;
    double b0;
    
    for(n=0; it != buffer.end()-1; ++it, ++n){
      _lineStream.clear();
      _lineStream.str(*it);
      
      _lineStream >> i >> j >> b0;
      
      if (_lineStream.fail() || ! _lineStream.eof()){
	io::messages.add("Bad line in SOLVENTCONSTR block",
			 "In_Topology", io::message::error);	
      }
      
      // the solvent (distance constraints) bond types
      // (K is set to 0.0)
      b.push_back(interaction::bond_type_struct(0, b0));
    }

    if (n!=num){
      io::messages.add("Error in SOLVENTCONSTR block (num != n)",
		       "In_Topology", io::message::error);
    }
  }
  
}

void io::In_Topology
::read_angles(std::vector<interaction::angle_type_struct> &b)
{
  
  DEBUG(10, "BONDANGLETYPE block");

  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  buffer = m_block["BONDANGLETYPE"];

  if (buffer.size()==0)
    io::messages.add("BONDANGLETYPE block not found!", "In_Topology",
		     io::message::error);

  // 1. BONDTYPE 2. number of types
  for (it = buffer.begin() + 2; 
       it != buffer.end() - 1; ++it) {

    double k, cos0;
    _lineStream.clear();
    _lineStream.str(*it);
      
    _lineStream >> k >> cos0;
      
    if (_lineStream.fail()){
      std::cout << *it << std::endl;
      io::messages.add("bad line in BONDANGLETYPE block", "In_Topology",
		       io::message::error);
    }
    if (! _lineStream.eof()){
      std::cout << *it << std::endl;
      io::messages.add("eof not reached in BONDANGLETYPE block",
		       "InTopology", io::message::warning);
    }

    // and add...
    b.push_back(interaction::angle_type_struct(k, cos(cos0 * 2 * math::Pi / 360.0)));
  }

}

void io::In_Topology
::read_improper_dihedrals(std::vector<interaction::improper_dihedral_type_struct> &i)
{
  
  DEBUG(10, "IMPDIHEDRALTYPE block");

  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  buffer = m_block["IMPDIHEDRALTYPE"];

  if (buffer.size()==0)
    io::messages.add("IMPDIHEDRALTYPE block not found!", "In_Topology",
		     io::message::error);

  // 1. IMPDIHEDRALTYPE 2. number of types
  for (it = buffer.begin() + 2; 
       it != buffer.end() - 1; ++it) {

    double k, q0;
    _lineStream.clear();
    _lineStream.str(*it);
      
    _lineStream >> k >> q0;
      
    if (_lineStream.fail()){
      std::cout << *it << std::endl;
      io::messages.add("bad line in IMPDIHEDRALTYPE block", "In_Topology",
		       io::message::error);
    }
    if (! _lineStream.eof()){
      std::cout << *it << std::endl;
      io::messages.add("eof not reached in IMPDIHEDRALTYPE block",
		       "InTopology", io::message::warning);
    }

    // and add...
    i.push_back(interaction::improper_dihedral_type_struct(k*180*180/math::Pi/math::Pi,
							   q0 * math::Pi / 180.0));
  }

}

void io::In_Topology
::read_dihedrals(std::vector<interaction::dihedral_type_struct> &d)
{
  
  DEBUG(10, "DIHEDRALTYPE block");

  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  buffer = m_block["DIHEDRALTYPE"];

  if (buffer.size()==0)
    io::messages.add("DIHEDRALTYPE block not found!", "In_Topology",
		     io::message::error);

  // 1. DIHEDRALTYPE 2. number of types
  for (it = buffer.begin() + 2; 
       it != buffer.end() - 1; ++it) {

    double k, pd;
    int m;

    _lineStream.clear();
    _lineStream.str(*it);
      
    _lineStream >> k >> pd >> m;
      
    if (_lineStream.fail()){
      std::cout << *it << std::endl;
      io::messages.add("bad line in DIHEDRALTYPE block", "In_Topology",
		       io::message::error);
    }
    if (! _lineStream.eof()){
      std::cout << *it << std::endl;
      io::messages.add("eof not reached in DIHEDRALTYPE block",
		       "InTopology", io::message::warning);
    }

    // and add...
    d.push_back(interaction::dihedral_type_struct(k, pd, m));
  }

}


void io::In_Topology
::read_lj_parameter(std::vector<std::vector
		    <interaction::lj_parameter_struct> > 
		    & lj_parameter)
{
  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;

  { // LJPARAMETERS
    
    DEBUG(10, "LJPARAMETERS block");
    
    buffer = m_block["LJPARAMETERS"];
    if (!buffer.size()){
      io::messages.add("No LJPARAMETERS block found in topology!",
		       "In_Topology",
		       io::message::error);
      return;
    }
    
    int num, n;
    
    it = buffer.begin() + 1;
    _lineStream.clear();
    _lineStream.str(*it);
    _lineStream >> num;
    
    // calculate the matrix size from: x = n*(n+1)/2
    unsigned int sz = unsigned(sqrt(double((8*num+1)-1))/2);

    lj_parameter.resize(sz);
    std::vector< std::vector<interaction::lj_parameter_struct> >::iterator
      lj_it = lj_parameter.begin(),
      lj_to = lj_parameter.end();
  
    for(; lj_it!=lj_to; ++lj_it)
      lj_it->resize(sz);
    
    ++it;
    
    for (n=0; it != buffer.end() - 1; ++it, ++n) {
      
      interaction::lj_parameter_struct s;
      int i, j;
      
      _lineStream.clear();
      _lineStream.str(*it);
      
      _lineStream >> i >> j >> s.c12 >> s.c6 >> s.cs12 >> s.cs6;

      --i;
      --j;
      
      if (_lineStream.fail() || ! _lineStream.eof())
	io::messages.add("bad line in LJPARAMETERS block", "In_Topology",
			 io::message::error);
      
      if (i >= int(sz) || j >= int(sz)){
	DEBUG(7, "wrong iac in LJPARAMETERS: i=" << i << " j=" << j
	      << " sz=" << sz);
	io::messages.add("wrong integer atom code in LJPARAMETERS block",
			 "In_Topology", 
			 io::message::error);
      }

      lj_parameter[i][j] = s;
      lj_parameter[j][i] = s;
      
    }

    if (num != n){
      io::messages.add("Reading the LJPARAMETERS failed (n != num)",
		       "InTopology",
		       io::message::error);
    }
  } // LJPARAMETER

}


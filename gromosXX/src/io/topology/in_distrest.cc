/**
 * @file in_distrest.cc
 * implements methods of In_Topology.
 */


#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <interaction/interaction_types.h>
#include <configuration/configuration.h>

#include <io/instream.h>
#include <io/blockinput.h>

#include "in_distrest.h"

#undef MODULE
#undef SUBMODULE
#define MODULE io
#define SUBMODULE topology

void 
io::In_Distrest::read(topology::Topology& topo,
		      configuration::Configuration & conf,
		      simulation::Simulation & sim){
  
  DEBUG(7, "reading in a distance restraints file");

  std::cout << "DISTRANCE RESTRAINTS\n";
  
  std::vector<std::string> buffer;
  std::vector<std::string>::const_iterator it;
  
  { // DISTREST
    DEBUG(10, "DISRESSPEC block");
    buffer = m_block["DISRESSPEC"];
    
    if (!buffer.size()){
      io::messages.add("no DISRESSPEC block in distance restraints file",
		       "in_distrest", io::message::error);
      return;
    }

    std::vector<std::string>::const_iterator it = buffer.begin()+1,
      to = buffer.end()-1;

    double dish,disc;
    int type1, type2;
    std::vector<int> atom1(4), atom2(4);
    double r0,w0;
    int rah;
    

    DEBUG(10, "reading in DISTREST data");

    switch(sim.param().distrest.distrest){
      case 0:
	std::cout << "\tDistance restraints OFF\n";
	// how did you get here?
	break;
      case 1:
	std::cout << "\tDistance restraints ON\n";
	
	break;
      case 2:
	std::cout << "\tDistance restraints ON\n"
		  << "\t\t(using force constant K*w0)\n";
	break;
	
    }


    _lineStream.clear();
    _lineStream.str(*it);

    _lineStream >> dish >> disc;

    ++it;
    std::cout << std::setw(10) << "DISH"
	      << std::setw(10) << "DISC"
	      << "\n" 
	      <<  std::setw(10)<< dish 
	      <<  std::setw(10)<< disc
	      << "\n";
    
    std::cout << std::setw(10) << "i"
	      << std::setw(8) << "j"
	      << std::setw(8) << "k"
	      << std::setw(8) << "l"
	      << std::setw(5) << "type"
	      << std::setw(10) << "i"
	      << std::setw(8) << "j"
	      << std::setw(8) << "k"
	      << std::setw(8) << "l"
	      << std::setw(5) << "type"
	      << std::setw(8) << "r0"
	      << std::setw(8) << "w0"
	      << std::setw(4) << "rah"
	      << "\n";
    
    
    for(int i=0; it != to; ++i, ++it){
      
      DEBUG(11, "\tnr " << i);
      
      _lineStream.clear();
      _lineStream.str(*it);

      _lineStream >> atom1[0] >> atom1[1] >> atom1[2] >>atom1[3] >> type1;
      _lineStream >> atom2[0] >> atom2[1] >> atom2[2] >>atom2[3] >> type2;
      _lineStream >> r0>> w0 >> rah;

    
      if(_lineStream.fail()){
	io::messages.add("bad line in DISTREST block",
			 "In_Distrest",
			 io::message::error);
      }
      
      for(int j=0; j<4; ++j)
	{
	  --atom1[j];
	  --atom2[j];
	}
      

      util::Virtual_Atom v1(type1, atom1, dish, disc);
      util::Virtual_Atom v2(type2, atom2, dish, disc);
    
      topo.distance_restraints().push_back
	(topology::distance_restraint_struct(v1,v2,r0,w0,rah));

      std::cout << std::setw(10) << atom1[0]+1
		<< std::setw(8) << atom1[1]+1
		<< std::setw(8) << atom1[2]+1
		<< std::setw(8) << atom1[3]+1
		<< std::setw(5) << type1
		<< std::setw(10) << atom2[0]+1
		<< std::setw(8) <<  atom2[1]+1
		<< std::setw(8) << atom2[2]+1
		<< std::setw(8) << atom2[3]+1
		<< std::setw(5) << type2
		<< std::setw(8) << r0
		<< std::setw(8) << w0
		<< std::setw(4) << rah
		<< "\n";
    }
    
  } // DISTREST
  
  { // PERTDISRESPEC DISTREST
    DEBUG(10, "PERTDISRESSPEC block");
    buffer = m_block["PERTDISRESSPEC"];
    
    if (!buffer.size()){
      
      return;
    }

    std::vector<std::string>::const_iterator it = buffer.begin()+1,
      to = buffer.end()-1;

    double dish,disc;
    int type1, type2;
    std::vector<int> atom1(4), atom2(4);
    double A_r0, B_r0,A_w0, B_w0;
    int rah;
    
    

    DEBUG(10, "reading in DISTREST (PERTDISRESSPEC data");

    switch(sim.param().distrest.distrest){
      case 0:
	std::cout << "\tPerturbed Distance restraints OFF\n";
	// how did you get here?
	break;
      case 1:
	std::cout << "\tPerturbed Distance restraints ON\n";
	
	break;
      case 2:
	std::cout << "\tPerturbed Distance restraints ON\n"
		  << "\t\t(using force constant K*w0)\n";
	break;
	
    }


    _lineStream.clear();
    _lineStream.str(*it);

    _lineStream >> dish >> disc;

    ++it;
    std::cout << std::setw(10) << "DISH"
	      << std::setw(10) << "DISC"
	      << "\n" 
	      <<  std::setw(10)<< dish 
	      <<  std::setw(10)<< disc
	      << "\n";
    
    std::cout << std::setw(10) << "i"
	      << std::setw(8) << "j"
	      << std::setw(8) << "k"
	      << std::setw(8) << "l"
	      << std::setw(5) << "type"
	      << std::setw(10) << "i"
	      << std::setw(8) << "j"
	      << std::setw(8) << "k"
	      << std::setw(8) << "l"
	      << std::setw(5) << "type"
	      << std::setw(8) << "A_r0"
	      << std::setw(8) << "A_w0"
	      << std::setw(8) << "B_r0"
	      << std::setw(8) << "B_w0"
	      << std::setw(4) << "rah"
	      << "\n";
    
    
    for(int i=0; it != to; ++i, ++it){
      
      DEBUG(11, "\tnr " << i);
      
      _lineStream.clear();
      _lineStream.str(*it);

      _lineStream >> atom1[0] >> atom1[1] >> atom1[2] >>atom1[3] >> type1;
      _lineStream >> atom2[0] >> atom2[1] >> atom2[2] >>atom2[3] >> type2;
      _lineStream >> A_r0 >> A_w0 >> B_r0 >> B_w0 >> rah;

    
      if(_lineStream.fail()){
	io::messages.add("bad line in PERTDISTREST block",
			 "In_Distrest",
			 io::message::error);
      }
      
      for(int j=0; j<4; ++j)
	{
	  --atom1[j];
	  --atom2[j];
	}
      

      util::Virtual_Atom v1(type1, atom1, dish, disc);
      util::Virtual_Atom v2(type2, atom2, dish, disc);
    
      topo.perturbed_distance_restraints().push_back
	(topology::perturbed_distance_restraint_struct(v1,v2,A_r0,B_r0,A_w0,B_w0, rah));

      std::cout << std::setw(10) << atom1[0]+1
		<< std::setw(8) << atom1[1]+1
		<< std::setw(8) << atom1[2]+1
		<< std::setw(8) << atom1[3]+1
		<< std::setw(5) << type1
		<< std::setw(10) << atom2[0]+1
		<< std::setw(8) <<  atom2[1]+1
		<< std::setw(8) << atom2[2]+1
		<< std::setw(8) << atom2[3]+1
		<< std::setw(5) << type2
		<< std::setw(8) << A_r0	    
		<< std::setw(8) << A_w0
		<< std::setw(8) << B_r0
		<< std::setw(8) << B_w0
		<< std::setw(8) << rah
		<< "\n";
    }//PERTDISRESPEC DISTREST
    
    std::cout << "END\n";
  
  }
  
}

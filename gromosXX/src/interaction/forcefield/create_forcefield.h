/**
 * @file create_forcefield.h
 */

#ifndef INCLUDED_CREATE_FORCEFIELD_H
#define INCLUDED_CREATE_FORCEFIELD_H

namespace interaction
{
  
  /**
   * create a Gromos96 (like) forcefield.
   */
  void create_g96_forcefield(interaction::Forcefield & ff,
			     topology::Topology const & topo,
			     simulation::Parameter const & param,
			     io::In_Topology & it);
    
  
}

#endif

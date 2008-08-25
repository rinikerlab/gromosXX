/**
 * @file in_perturbation.h
 * read in a perturbation topology file (03 format)
 */

/**
 * @page pttopo perturbation topology format
 * @date 11-06-2008
 *
 * - @ref just
 * - @ref title
 * - @ref scaled
 *
 *<ul>
<li><a href="#title">title</a></li>
<li><a href="#scaled">scaled interactions</a></li>
<li><a href="#pertatom">atoms</a></li>
<li><a href="#pertatompair">atom pairs</a></li>
<li><a href="#pertond">bonds</a></li>
<li><a href="#pertangle">angles</a></li>
<li><a href="#pertimp">improper dihedral angles</a></li>
<li><a href="#pertdih">dihedral angles</a></li>
<li><a href="#pertpolparam">Perturbed polarizable atoms</a></li>
<li><a href="#edsatomparam">EDS-perturbed atoms</a></li>
</ul>

@section title TITLE block
@verbatim
TITLE
  Example Perturbation of Diethylstilbestrol (DES) into Estradiol (E2)
  22-2-2000, Chris Oostenbrink
END
@endverbatim

@section scaled Scaled Interactions
Specific (nonbonded) interactions can be scaled dependent on lambda.
@verbatim
SCALEDINTERACTIONS
# number of scaled interactions
  1
# scale nonbonded interaction between atoms of energy group 1 and
# atoms of energy group 2:
# from 0.5 (at lambda = 0) to 0.1 (at lambda = 1)
  1     2      0.5     0.1
END
@endverbatim

<h2><a name="pertatomparam">Atoms</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTATOMPARAM
# number of perturbed atoms
   2
#  
#   NR RES NAME IAC(A) MASS(A)  CHARGE(A) IAC(B) MASS(B) CHARGE(B)   ALJ  ACRF
     3   1  CZ1     11  12.011       0.15     12  16.011      0.25   1.0  1.0
    17   1  CB2     14  15.035       0.00     13  10.035      0.10   0.0  0.0
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertatompair">Atom Pairs</a></h2>
Interactions might change from excluded to 1,4 or to not excluded
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTATOMPAIR
# number of perturbed atom pairs
   1
#  interaction:
#    0 : excluded
#    1 : normal interaction
#    2 : 1,4 interaction
#
#  NR(I) NR(J) INTERACTION(A) INTERACTION(B)
    2     5    2              1
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertbondstretch">Bonds</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTBONDSTRETCH
# number of perturbed bonds
	3
#   atom(i) atom(j) bond_type(A) bond_type(B)
       4       6           15	        26
       6      12           15   	25
       3       8           15   	25
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertbondstretchh">H Bonds</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTBONDSTRETCHH
# number of perturbed hydrogen bonds
        3
#   atom(i) atom(j) bond_type(A) bond_type(B)
       4       6           15           26
       6      12           15           25
       3       8           15           25
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertbondangle">Bond Angles</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTBONDANGLE
# number of perturbed bond angles
    3
#    atom(i) atom(j) atom(k) type(A) type(B)
        2       3       8      26       8
        4       6      12      26       7
        3       8      10      26       7
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertbondangleh">H Bond Angles</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTBONDANGLEH
# number of perturbed hydrogen bond angles
    3
#    atom(i) atom(j) atom(k) type(A) type(B)
        2       3       8      26       8
        4       6      12      26       7
        3       8      10      26       7
END
</PRE></CODE></td></tr></tbody>
</TABLE>


<h2><a name="pertimproperdih">Improper Dihedral Angles</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTIMPROPERDIH
# number of perturbed improper dihedrals
    2
#    atom(i) atom(j) atom(k) atom(l)  type(A) type(B)
       12      13      10       6        1       2
       18      19      13      16        1       2
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertimproperdihh">H Improper Dihedral Angles</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTIMPROPERDIHH
# number of perturbed hydrogen improper dihedrals
    2
#    atom(i) atom(j) atom(k) atom(l)  type(A) type(B)
       12      13      10       6        1       2
       18      19      13      16        1       2
END
</PRE></CODE></td></tr></tbody>
</TABLE>


<h2><a name="pertproperdih">Dihedral Angles</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTPROPERDIH
# number of perturbed dihedrals
    2
#    atom(i) atom(j) atom(k) atom(l)  type(A) type(B)
	6      12      13      14        1      17
       12      13      18      19        4      17
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertproperdihh">H Dihedral Angles</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTPROPERDIHH
# number of perturbed hydrogen dihedrals
    2
#    atom(i) atom(j) atom(k) atom(l)  type(A) type(B)
        6      12      13      14        1      17
       12      13      18      19        4      17
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="pertpolparam">Perturbed polarizable atoms</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
PERTPOLPARAM
# number of perturbed polarizable atoms 
# (atoms must also appear in PERTATOMPARAM)
  1
#   NR RES NAME   ALPHA(A) E_0(A)   ALPHA(B) E_0(B)
     1  1   OW    0.00093  80.0     0.00093  80.0
END
</PRE></CODE></td></tr></tbody>
</TABLE>

<h2><a name="edsatomparam">EDS-perturbed atoms</a></h2>
<table cellpadding="2" cellspacing="2" border="0" width="600"
 bgcolor="lightgrey"><tbody><tr><td valign="top"><CODE><PRE>
EDSATOMPARAM
# number of eds-perturbed atoms 
  6
# NR res name mass iac 0...N charge 0...N
# needed information is only: NR, IAC 0...N, charge 0...N
# NR    RES  NAME  IAC[0...N]  CHARGE[0...N]
# first water
  1      1    OW   4  19       -0.82  0     
  2      1    HW1  18 19        0.41  0     
  3      1    HW2  18 19        0.41  0    
# second water
  4      2    OW   19 4         0    -0.82  
  5      2    HW1  19 18        0     0.41 
  6      2    HW2  19 18        0     0.41 
END
</PRE></CODE></td></tr></tbody>
</TABLE>
 */

#ifndef INCLUDED_IN_PERTURBATION_H
#define INCLUDED_IN_PERTURBATION_H

#include <gromosXX/io/instream.h>

namespace io
{
  /**
   * @class In_Perturbation
   * reads in a perturbation topology file (03 version)
   * and parses it into Topology
   * @sa topology::Topology
   */
  class In_Perturbation : public GInStream
  {
  public:
    /**
     * Constructor.
     */
    In_Perturbation(std::istream &is);
    /**
     * parse the topology.
     */
    void read(topology::Topology &topo, simulation::Parameter &param);
    
  };
  
} // io

#endif

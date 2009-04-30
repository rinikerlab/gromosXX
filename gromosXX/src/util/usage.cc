/**
 * @file usage.cc
 * get usage string
 */

#include <stdheader.h>
#include "usage.h"

void util::get_usage(util::Known const &knowns, std::string &usage, std::string name)
{
  usage = "#\n# " + name + "\n\n";

  if (knowns.count("topo")){
    usage += "\t# topology data\n";
    usage += "\t@topo        filename\n\n";
  }

  if (knowns.count("cg_topo")){
    usage += "\t# coarse-grained topology data\n";
    usage += "\t@cg_topo     filename\n\n";
  }
  
  if (knowns.count("pttopo")){
    usage += "\t# perturbation topology data\n";
    usage += "\t# @pttopo    filename\n\n";
  }

  if (knowns.count("cg_pttopo")){
    usage += "\t# coarse-grained perturbation topology data\n";
    usage += "\t# @cg_pttopo filename\n\n";
  }
  
  if (knowns.count("conf")){
    usage += "\t# coordinates\n";
    usage += "\t@conf        filename\n\n";
  }

  if (knowns.count("cg_conf")){
    usage += "\t# coarse-grained coordinates\n";
    usage += "\t@cg_conf     filename\n\n";
  }
  
  if (knowns.count("input")){
    usage += "\t# input parameter\n";
    usage += "\t@input       filename\n\n";
  }

  if (knowns.count("cg_input")){
    usage += "\t# coarse-grained input parameter\n";
    usage += "\t@cg_input    filename\n\n";
  }

  if (knowns.count("fin")){
    usage += "\t# output finale coordinates\n";
    usage += "\t@fin         filename\n\n";
  }

  if (knowns.count("cg_fin")){
    usage += "\t# coarse-grained output finale coordinates\n";
    usage += "\t@cg_fin      filename\n\n";
  }
  
  if (knowns.count("trj")){
    usage += "\t# output coordinates trajectory\n";
    usage += "\t@trj         filename\n\n";
  }

  if (knowns.count("cg_trj")){
    usage += "\t# coarse-grained output coordinates trajectory\n";
    usage += "\t@cg_trj      filename\n\n";
  }
  
  if (knowns.count("trv")){
    usage += "\t# output velocity trajectory\n";
    usage += "\t# @trv       filename\n\n";
  }
  
  if (knowns.count("trf")){
    usage += "\t# output force trajectory\n";
    usage += "\t# @trf       filename\n\n";
  }

  if (knowns.count("trs")){
    usage += "\t# output special trajectory\n";
    usage += "\t# @trs       filename\n\n";
  }  
  
  if (knowns.count("tramd")){
    usage += "\t# output RAMD trajectory\n";
    usage += "\t# @tramd     filename\n\n";
  }
  
  if (knowns.count("tre")){
    usage += "\t# output energy trajectory\n";
    usage += "\t# @tre       filename\n\n";
  }

  if (knowns.count("cg_tre")){
    usage += "\t# output coarse-grained energy trajectory\n";
    usage += "\t# @cg_tre    filename\n\n";
  }

  if (knowns.count("re")){
    usage += "\t# output replica energy trajectory (per switch)\n";
    usage += "\t# @re        filename\n\n";
  }
  
  if (knowns.count("bae")){
    usage += "\t# output block averaged energy trajectory\n";
    usage += "\t# @bae       filename\n\n";
  }
  
  if (knowns.count("trg")){
    usage += "\t# output free energy trajectory\n";
    usage += "\t# @trg       filename\n\n";
  }
  
  if (knowns.count("bag")){
    usage += "\t# output block averaged free energy trajectory\n";
    usage += "\t# @bag       filename\n\n";    
  }
  
  if (knowns.count("posresspec")){
    usage += "\t# position restraints specification\n";
    usage += "\t# @posresspec    filename\n\n";
  }
  
  if (knowns.count("refpos")){
    usage += "\t# position restraints\n";
    usage += "\t# @refpos    filename\n\n";
  }
  
  if (knowns.count("distrest")){
    usage += "\t# distance restraints specification\n";
    usage += "\t# @distrest  filename\n\n";
  }

  if (knowns.count("dihrest")){
    usage += "\t# dihedral restraints specification\n";
    usage += "\t# @dihrest  filename\n\n";
  }
  
  if (knowns.count("jval")){
    usage += "\t# J-value restraints specification\n";
    usage += "\t# @jval      filename\n\n";
  }

  if (knowns.count("xray")){
    usage += "\t# X-ray restraints specification\n";
    usage += "\t# @xray      filename\n\n";
  }

  if (knowns.count("lud")){
    usage += "\t# local elevation umbrella database file\n";
    usage += "\t# @lud       filename\n\n";
  }

  if (knowns.count("led")){
    usage += "\t# local elevation coordinate definition file\n";
    usage += "\t# @led       filename\n\n";
  }
  
  if (knowns.count("friction")){
    usage += "\t# atomic friction coefficients\n";
    usage += "\t# @friction   filename\n\n";
  }
  
  if (knowns.count("master")){
    usage += "\t# replica exchange: master process\n";
    usage += "\t# @master    name\n\n";
  }

  if (knowns.count("slave")){
    usage += "\t# replica exchange: slave process\n";
    usage += "\t# @slave     name\n\n";
  }
  
  if (knowns.count("print")){
    usage += "\t# print additional information\n";
    usage += "\t# @print     <pairlist/force>\n\n";
  }
  
  if (knowns.count("trp")){
    usage += "\t# write additional information to file\n";
    usage += "\t# @trp       filename\n\n";
  }
  
  if (knowns.count("anatrj")){
    usage += "\t# re-analyze trajectory\n";
    usage += "\t# @anatrj    filename\n\n";
  }
  
  if (knowns.count("verb")){
    usage += "\t# control verbosity (in debug builds)\n";
    usage += "\t# @verb      <[module:][submodule:]level>\n\n";
  }
  
  if (knowns.count("version")){
    usage += "\t# print version information\n";
    usage += "\t# @version\n\n";
  }
  // usage += "#\n\n";

}



/**
 * @file in_parameter.cc
 * implements methods of In_Parameter
 */

#include <util/stdheader.h>

#include <topology/core/core.h>
#include <topology/topology.h>

#include <simulation/multibath.h>
#include <simulation/parameter.h>

#include <io/instream.h>
#include <io/blockinput.h>

#include "in_parameter.h"

#undef MODULE
#define MODULE io
#undef SUBMODULE
#define SUBMODULE parameter

static std::set<std::string> block_read;

/**
 * Store standard parameters in the Parameter
 */
void io::In_Parameter::read(simulation::Parameter &param)
{
  DEBUG(7, "reading input");

  read_SYSTEM(param);
  read_START(param); // and CENTREOFMASS
  read_STEP(param);
  read_BOUNDARY(param);
  read_SUBMOLECULES(param);
  read_MULTIBATH(param);
  read_PCOUPLE(param);
  read_PRINT(param);
  read_WRITE(param);
  read_SHAKE(param);
  read_FORCE(param); // and FORCEFIELD
  read_PLIST(param);
  read_LONGRANGE(param);
  read_POSREST(param);
  read_PERTURB(param);
  
  DEBUG(7, "input read...");

  for(std::map<std::string, std::vector<std::string> >::const_iterator
	it = m_block.begin(),
	to = m_block.end();
      it != to;
      ++it){

    if (block_read.count(it->first) == 0 && it->second.size()){
      io::messages.add("block " + it->first + " not supported!",
		       "In_Parameter",
		       io::message::warning);
    }
  }
}

/**
 * read the SYSTEM block.
 */
void io::In_Parameter::read_SYSTEM(simulation::Parameter &param)
{
  DEBUG(8, "reading SYSTEM");

  std::vector<std::string> buffer;
  buffer = m_block["SYSTEM"];
  std::string s;
  
  if (!buffer.size()){
    io::messages.add("no SYSTEM block in input", "In_Parameter", io::message::error);
    param.system.nsm = 0;
    param.system.npm = 0;
    return;
  }

  block_read.insert("SYSTEM");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  _lineStream >> param.system.npm 
	      >> param.system.nsm;
 
  if (_lineStream.fail())
    io::messages.add("bad line in SYSTEM block",
		       "In_Parameter", io::message::error);

  /**
  if (!_lineStream.eof())
    io::messages.add("End of line not reached in SYSTEM block, but should have been: \n" + _lineStream.str() +  "\n",
		       "In_Parameter", io::message::warning);
  */
  
  // we might need to also allow for 0...
  if (param.system.npm != 1)
    io::messages.add("SYSTEM: currently only NPM=1 allowed",
		     "io::In_Parameter::read_SYSTEM",
		     io::message::error);
  if(param.system.nsm < 0)
    io::messages.add("SYSTEM: NSM should be >0",
		     "io::In_Parameter::read_SYSTEM",
		     io::message::error);

} 

/**
 * read the STEP block.
 */
void io::In_Parameter::read_STEP(simulation::Parameter &param)
{
  DEBUG(8, "read STEP");
  
  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["STEP"];
  
  if (!buffer.size()){
    io::messages.add("no STEP block in input", "In_Parameter", io::message::error);
    param.step.number_of_steps = 0;
    param.step.t0 = 0;
    param.step.dt = 0;
    return;
  }

  block_read.insert("STEP");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  _lineStream >> param.step.number_of_steps
	      >> param.step.t0
	      >> param.step.dt;
  
  if (_lineStream.fail())
    io::messages.add("bad line in STEP block",
		       "In_Parameter", io::message::error);

  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached in STEP block, but should have been: \n" + s +  "\n",
		       "In_Parameter", io::message::warning);
  */

  if(param.step.t0 < 0)
    io::messages.add("Negative time in STEP block is not supported",
		     "In_Parameter", io::message::error);
  if(param.step.number_of_steps <= 0)
    io::messages.add("We want to do at least one step...",
		     "In_Parameter", io::message::error);
}

/**
 * the SHAKE block.
 */
void io::In_Parameter::read_SHAKE(simulation::Parameter &param)
{
  DEBUG(8, "read SHAKE");

  std::vector<std::string> buffer;
  std::string s, sntc;
  
  buffer = m_block["SHAKE"];

  if (!buffer.size()){
    io::messages.add("no SHAKE block", "In_Parameter",io::messages.notice);
    param.shake.ntc = 1;
    param.shake.tolerance = 0;
    return;
  }
  
  block_read.insert("SHAKE");
  
  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

  _lineStream >> sntc
	      >> param.shake.tolerance;
  
  if (_lineStream.fail())
    io::messages.add("bad line in SHAKE block",
		       "In_Parameter", io::message::error);

  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached in SHAKE block, but should have been: \n" + s +  "\n",
		       "In_Parameter", io::message::warning);
  */

  std::transform(sntc.begin(), sntc.end(), sntc.begin(), tolower);

  if(sntc=="solvent") param.shake.ntc=1;
  else if(sntc=="hydrogen") param.shake.ntc=2;
  else if(sntc=="all") param.shake.ntc=3;
  else if(sntc=="specified") param.shake.ntc=4;
  else param.shake.ntc=atoi(sntc.c_str());

  if(errno)
    io::messages.add("NTC not understood in SHAKE block",
		       "In_Parameter", io::message::error);
  if(param.shake.ntc<1 || param.shake.ntc > 4)
    io::messages.add("NTC not understood in SHAKE block",
		       "In_Parameter", io::message::error);
  if(param.shake.tolerance<=0.0)
    io::messages.add("tolerance in SHAKE block should be > 0",
		       "In_Parameter", io::message::error);
}

/**
 * the FLEXCON block.
 */
/*
void io::In_Parameter::read_FLEXCON(int &lfcon)
{
  std::vector<std::string> buffer;
  
  buffer = m_block["FLEXCON"];

  if (!buffer.size()){
    io::messages.add("no FLEXCON block", "In_Parameter",io::messages.notice);
    lfcon = 0;
    return;
  }
  
  it = buffer.begin() + 1;
  _lineStream.clear();
  _lineStream.str(s);
  
  _lineStream >> lfcon;
  
  if (_lineStream.fail())
    io::messages.add("bad line in FLEXCON block",
		       "In_Parameter", io::message::error);

  if (!_lineStream.eof())
    io::messages.add("end of line not reached in FLEXCON block,"
		     " but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);

}
*/ 
/**
 * read the PRINT
 */
void io::In_Parameter::read_PRINT(simulation::Parameter &param)
{
  DEBUG(8, "read PRINT");

  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["PRINT"];

  if (!buffer.size()){
    io::messages.add("no PRINT block", "In_Parameter", io::message::notice);
    param.print.stepblock = 1;
    param.print.centreofmass = 1;
    param.print.monitor_dihedrals = false;
    return;
  }

  block_read.insert("PRINT");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

  _lineStream >> param.print.stepblock
	      >> param.print.centreofmass
	      >> param.print.monitor_dihedrals;
  
  if (_lineStream.fail())
    io::messages.add("bad line in PRINT block",
		     "In_Parameter", io::message::error);

  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached in PRINT, but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);
  */
  if(param.print.stepblock<=0)
    io::messages.add("PRINT block: print stepblock should be >0",
		     "In_Parameter", io::message::error);
  if(param.print.centreofmass<=0)
    io::messages.add("PRINT block: print centre of mass should be >0",
		     "In_Parameter", io::message::error);
  if( param.print.monitor_dihedrals)
    io::messages.add("Dihedral angle monitoring currently not supported",
		     "In_Parameter", io::message::warning);
}

/**
 * read the WRITE
 */
void io::In_Parameter::read_WRITE(simulation::Parameter &param)
{
  DEBUG(8, "read WRITE");

  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["WRITE"];

  if (!buffer.size()){
    io::messages.add("no WRITE block", "In_Parameter", io::message::notice);
    param.write.position = 1;
    param.write.velocity = 1;
    param.write.energy = 1;
    param.write.free_energy = 1;
    return;
  }

  block_read.insert("WRITE");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

  int ntwse, ntwp;
  
  _lineStream >> param.write.position
	      >> ntwse
	      >> param.write.velocity
	      >> param.write.energy
	      >> param.write.free_energy
	      >> ntwp;

  
  if (_lineStream.fail())
    io::messages.add("bad line in WRITE block",
		     "In_Parameter", io::message::error);
  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached in WRITE block, "
		     "but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);
  */
  if(ntwse!=0)
    io::messages.add("NTWSE != 0 not supported",
		     "In_Parameter", io::message::error);
  if(ntwp!=1)
    io::messages.add("NTWP != 0 not supported",
		     "In_Parameter", io::message::error);
  if(param.write.position < 0)
    io::messages.add("WRITE block: NTWX should be >= 0",
		     "In_Parameter", io::message::error);
   if(param.write.velocity < 0)
    io::messages.add("WRITE block: NTWV should be >= 0",
		     "In_Parameter", io::message::error);
   if(param.write.energy < 0)
    io::messages.add("WRITE block: NTWE should be >= 0",
		     "In_Parameter", io::message::error);
   if(param.write.free_energy < 0)
    io::messages.add("WRITE block: NTWG should be >= 0",
		     "In_Parameter", io::message::error);
}

/**
 * read the PCOUPLE block.
 */
void io::In_Parameter::read_PCOUPLE(simulation::Parameter &param)
{
  DEBUG(8, "read PCOUPLE");

  std::vector<std::string> buffer;
  std::string s;
  

  // first try for a PCOUPLE03 block
  buffer = m_block["PCOUPLE03"];

  if (buffer.size()){

    block_read.insert("PCOUPLE03");

    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
    std::string s1, s2, s3;
    
    _lineStream >> s1 >> s2 
		>> param.pcouple.compressibility 
		>> param.pcouple.tau 
		>> s3;

    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
	_lineStream >> param.pcouple.pres0(i, j);
      }
    }

    if (_lineStream.fail())
      io::messages.add("bad line in PCOUPLE03 block",
		       "In_Parameter", io::message::error);
    
    std::transform(s1.begin(), s1.end(), s1.begin(), tolower);
    std::transform(s2.begin(), s2.end(), s2.begin(), tolower);
    std::transform(s3.begin(), s3.end(), s3.begin(), tolower);

    if (s1 == "off"){
      param.pcouple.calculate = false;
    } 
    else if (s1 == "calc"){
      param.pcouple.calculate = true;
    }
    else if (s1 == "scale"){
      param.pcouple.calculate = true;
    }
    else{
      io::messages.add("bad value for calc switch in PCOUPLE03 block\n"
		       "(off,calc,scale)",
		       "In_Parameter", io::message::error);
      param.pcouple.calculate = false;
    }
  
    if (s2 == "off")
      param.pcouple.scale = math::pcouple_off;
    else if (s2 == "iso")
      param.pcouple.scale = math::pcouple_isotropic;
    else if (s2 == "aniso")
      param.pcouple.scale = math::pcouple_anisotropic;
    else if (s2 == "full")
      param.pcouple.scale = math::pcouple_full_anisotropic;
    else{
      io::messages.add("bad value for SCALE switch in PCOUPLE03 block\n"
		       "(off,iso,aniso,full)",
		       "In_Parameter", io::message::error);
	param.pcouple.scale = math::pcouple_off;
    }
    if (s3 == "none")
      param.pcouple.virial = math::no_virial;
    else if (s3 == "atomic")
      param.pcouple.virial = math::atomic_virial;
    else if (s3 == "molecular")
      param.pcouple.virial = math::molecular_virial;
    else{
      io::messages.add("bad value for virial switch in PCOUPLE03 block\n"
		       "(none,atomic,molecular)",
		       "In_Parameter", io::message::error);
      param.pcouple.virial = math::no_virial;
    }
  } // PCOUPLE03 block
  else{
    
    buffer = m_block["PCOUPLE"];
    if (!buffer.size()){
      // everything should be initialized to nothing
      return;
    }

    block_read.insert("PCOUPLE");
 
    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

    int ntp;    
    double p0;
    
    _lineStream >> ntp >> p0 
		>> param.pcouple.compressibility 
		>> param.pcouple.tau;
    
    if (_lineStream.fail())
      io::messages.add("bad line in PCOUPLE block",
		       "In_Parameter", io::message::error);

    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
	if (i == j)
	  param.pcouple.pres0(i,j) = p0;
	else param.pcouple.pres0(i,j) = 0.0;
      }
    }
    if(ntp==1){
      param.pcouple.scale = math::pcouple_isotropic;
      param.pcouple.virial = math::molecular_virial;
    }
    else if(ntp==2){
      param.pcouple.scale = math::pcouple_anisotropic;
      param.pcouple.virial = math::molecular_virial;
    }
    else if(ntp>0 || ntp <0)
      io::messages.add("PCOUPLE block: illegal value for ntp (0,1,2)",
		       "In_Parameter",
		       io::message::error);
  }
  if (param.pcouple.calculate==false && param.pcouple.scale!=math::pcouple_off)
    io::messages.add("pressure coupling activated but "
		     "not calculating pressure",
		     "In_Parameter",
		     io::message::error);
  if (param.pcouple.calculate == true && param.pcouple.virial == math::no_virial)
    io::messages.add("PCOUPLE03 block: pressure calculation requested but"
		     " no virial specified!", "In_Parameter",
		     io::message::error);
  if(param.pcouple.compressibility <=0)
    io::messages.add("PCOUPLE block: compressibility should be >0 ",
		     "In_Parameter", io::message::error);
  if(param.pcouple.tau <=0)
    io::messages.add("PCOUPLE block: tau should be >0 ",
		     "In_Parameter", io::message::error);
}

/**
 * read the BOUNDARY block.
 */
void io::In_Parameter::read_BOUNDARY(simulation::Parameter &param)
{
  DEBUG(8, "read BOUNDARY");

  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["BOUNDARY"];
  
  if (!buffer.size()){
    io::messages.add("no BOUNDARY block", "In_Parameter", io::message::error);
    param.boundary.boundary = math::vacuum;
    return;
  }

  block_read.insert("BOUNDARY");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

  std::string ntb;
  int n, nrdbox;
  double b1, b2, b3, beta;

  _lineStream >> ntb >> b1 >> b2 >> b3 >> beta >> nrdbox;
  
  if (_lineStream.fail())
    io::messages.add("bad line in BOUNDARY block",
		     "In_Parameter", io::message::error);

  std::transform(ntb.begin(), ntb.end(), ntb.begin(), tolower);
  if(ntb=="vacuum") param.boundary.boundary=math::vacuum;
  else if(ntb=="rectangular") param.boundary.boundary=math::rectangular;
  else if(ntb=="triclinic") param.boundary.boundary=math::triclinic;
  else {
    n=atoi(ntb.c_str());
    if(errno){
      io::messages.add("wrong value for NTB in BOUNDARY block\n"
		       "vacuum, rectangular, triclinic, 0, +/-1, +/-2",
		       "In_Parameter", io::message::error);
      param.boundary.boundary=math::vacuum;
      return;
    }
    if(n==0) param.boundary.boundary=math::vacuum;
    else if(n>0) param.boundary.boundary=math::rectangular;
    else param.boundary.boundary=math::triclinic;
    
    if(abs(n)==2){
      param.pcouple.calculate=true;
      if(param.pcouple.virial==math::no_virial)
	param.pcouple.virial=math::molecular_virial;
    }
  }

  if(!nrdbox && param.boundary.boundary != math::vacuum){
    io::messages.add("Illegal value for NRDBOX in BOUNDARY block (should be 1)",
		     "In_Parameter", io::message::warning);
  }
}

/**
 * read the PERTURB block.
 */
void io::In_Parameter::read_PERTURB(simulation::Parameter &param)
{
  DEBUG(8, "read PERTURB");

  std::vector<std::string> buffer;
  std::string s;
  
  // try the new PERTURB03 block
  buffer = m_block["PERTURB03"];
  if (buffer.size()){

    block_read.insert("PERTURB03");


    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

    std::string b, s1, s2;
    _lineStream >> s1 
		>> param.perturbation.lambda
		>> param.perturbation.dlamt
		>> param.perturbation.lambda_exponent
		>> s2;
    
    if (_lineStream.fail())
      io::messages.add("bad line in PERTURB block",
		       "In_Parameter", io::message::error);
    
    std::transform(s1.begin(), s1.end(), s1.begin(), tolower);
    std::transform(s2.begin(), s2.end(), s2.begin(), tolower);
    
    if (s1 == "on") param.perturbation.perturbation = true;
    else if (s1 == "off") param.perturbation.perturbation = false;
    else {
      param.perturbation.perturbation = (atoi(s1.c_str())==1);
      if(errno){
	io::messages.add("bad value for NTG in PERTURB block\n"
			 "on,off,0,1",
			 "In_Parameter", io::message::error);
	param.perturbation.perturbation=false;
	return;
      }
    }
    
    if (s2 == "on") param.perturbation.scaling = true;
    else if (s2 == "off") param.perturbation.scaling = false;
    else {
      param.perturbation.scaling = (atoi(s1.c_str())==1);
      if(errno){
	io::messages.add("bad value for SCALING in PERTURB block\n"
			 "on,off,0,1",
			 "In_Parameter", io::message::error);
	param.perturbation.scaling=false;
	return;
      }
    }
  }
  else{
    // a common PERTURB block...
    buffer = m_block["PERTURB"];
    if (!buffer.size()){
      param.perturbation.perturbation=false;
      param.perturbation.lambda = 0;
      param.perturbation.lambda_exponent = 1;
      param.perturbation.dlamt = 0;
      param.perturbation.scaling = false;
      return;
    }

    block_read.insert("PERTURB");

    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

    int ntg, nrdgl;
    double dmu, dmut;
    double mmu;
    double alpha_lj, alpha_crf;
    
    _lineStream >> ntg >> nrdgl 
		>> param.perturbation.lambda 
		>> param.perturbation.dlamt 
		>> dmu >> dmut
		>> alpha_lj >> alpha_crf 
		>> param.perturbation.lambda_exponent
		>> mmu;
    
    param.perturbation.scaling = false;
    
    if (_lineStream.fail())
      io::messages.add("bad line in PERTURB block",
		       "In_Parameter", io::message::error);
    
    if (nrdgl)
      io::messages.add("PERTURB: nrdgl != 0 not allowed",
		       "In_Parameter", io::message::error);
    
    if (alpha_lj || alpha_crf){
      io::messages.add("PERTURB: softness constants taken from topology!",
		       "In_Parameter", io::message::notice);
    }
    
    param.perturbation.perturbation=(ntg!=0);
    
    if (ntg != 0 && ntg != 1)
      io::messages.add("PERTURB: only ntg = 0 or ntg = 1 allowed",
		       "In_Parameter", io::message::error);
  }
  
  if (param.perturbation.lambda_exponent<=0){
    io::messages.add("PERTURB: nlam > 0",
		     "In_Parameter", io::message::error);
  }
}

/**
 * read FORCE block.
 */
void io::In_Parameter::read_FORCE(simulation::Parameter &param)
{
  DEBUG(8, "read FORCE");

  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["FORCE"];

  if (!buffer.size()){
    io::messages.add("no FORCE block", "In_Parameter", io::message::error);
    param.force.bond = 0;
    param.force.angle = 0;
    param.force.improper = 0;
    param.force.dihedral = 0;
    param.force.nonbonded = 0;
    return;
  }

  block_read.insert("FORCE");
  

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  int bondH, angleH, impH, dihedralH, charge;
  size_t num, e, old_e=0;
  
  _lineStream >> bondH >> param.force.bond >> angleH >> param.force.angle
	      >> impH >> param.force.improper >> dihedralH >> param.force.dihedral
	      >> charge >> param.force.nonbonded;
  _lineStream >> num;
  if(num<=0){
    io::messages.add("number of energy groups in FORCE block should be > 0",
		     "In_Parameter", io::message::error);
    return;
  }
  
  for(size_t i=0; i<num; ++i){
    _lineStream >> e;
    param.force.energy_group.push_back(e-1);
    if(e<=old_e){
      io::messages.add("energy groups are not in order in FORCE block",
		       "In_Parameter", io::message::error);
      return;
    }
    old_e = e;
  }
  
  if (_lineStream.fail())
    io::messages.add("bad line in ENERGYGROUP (FORCE) block",
		     "In_Parameter", io::message::error);
  
  
  if (bondH ^ param.force.bond)
    io::messages.add("Force switch for bond and bond H has to be equal",
		     "In_Parameter", io::message::error);

  if (angleH ^ param.force.angle)
    io::messages.add("Force switch for angle and angle H has to be equal",
		     "In_Parameter", io::message::error);

  if (impH ^ param.force.improper)
    io::messages.add("Force switch for improper and improper H has to be equal",
		     "In_Parameter", io::message::error);

  if (dihedralH ^ param.force.dihedral)
    io::messages.add("Force switch for dihedral and dihedral H has to be equal",
		     "In_Parameter", io::message::error);

  if (charge ^ param.force.nonbonded)
    io::messages.add("Force switch for lj and charge has to be equal",
		     "In_Parameter", io::message::error);

  if (_lineStream.fail())
    io::messages.add("bad line in FORCE block",
		       "In_Parameter", io::message::error);

  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached in FORCE block, but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);
  */
  //from here read the force field block
  read_FORCEFIELD(param);
  
  if(param.force.bond < 0 || param.force.bond > 2)
    io::messages.add("Illegal value for force switch for bond",
		     "In_Parameter", io::message::error);
  if(param.force.bond < 0 || param.force.angle > 2)
    io::messages.add("Illegal value for force switch for angle",
		     "In_Parameter", io::message::error);
  if(param.force.bond < 0 || param.force.angle == 2)
    io::messages.add("Force switch for angle = 2 currently not implemented",
		     "In_Parameter", io::message::error);
  if(param.force.bond < 0 || param.force.improper > 1)
    io::messages.add("Illegal value for force switch for improper dihedral",
		     "In_Parameter", io::message::error);
  if(param.force.bond < 0 || param.force.dihedral > 1)
    io::messages.add("Illegal value for force switch for dihedral",
		     "In_Parameter", io::message::error);
  if(param.force.bond < 0 || param.force.nonbonded > 1)
    io::messages.add("Illegal value for force switch for nonbonded",
		     "In_Parameter", io::message::error);
}

/**
 * read FORCEFIELD block.
 */
void io::In_Parameter::read_FORCEFIELD(simulation::Parameter &param)
{
  DEBUG(8, "read FORCEFIELD");

  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["FORCEFIELD"];

  if (!buffer.size()){
    return;
  }
  
  block_read.insert("FORCEFIELD");

  io::messages.add("using FORCEFIELD block to determine bond term",
		   "In_Parameter", io::message::notice);
  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  _lineStream >> param.force.bond 
	      >> param.force.angle;
  
  if (_lineStream.fail())
    io::messages.add("bad line in FORCEFIELD block",
		     "In_Parameter", io::message::error);
  
  /*
  if (!_lineStream.eof())
    io::messages.add("end of line not reached in FORCEFIELD block,"
		     " but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);
  */
}

/**
 * read START block.
 */
void io::In_Parameter::read_START(simulation::Parameter &param)
{
  DEBUG(8, "read START");

  std::vector<std::string> buffer;
  std::string s;
  
  buffer = m_block["START"];
  
  if (!buffer.size()){
    io::messages.add("no START block", "In_Parameter", io::message::error);
    param.start.shake_pos = false;
    param.start.shake_vel = false;
    param.start.remove_com = false;
    param.start.generate_velocities = false;
    param.start.ig = 19536976;
    param.start.tempi = 0.0;
    return;
  }

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  block_read.insert("START");

  int ntx, init, ntx0;
  double heat;
  
  _lineStream >> ntx >> init 
	      >> param.start.ig 
	      >> param.start.tempi >> heat >> ntx0 
	      >> math::k_Boltzmann;
  
  if (_lineStream.fail())
    io::messages.add("bad line in START block",
		     "In_Parameter", io::message::error);
  
  read_CENTREOFMASS(param);
  
  if(ntx==1 || param.start.tempi)
    param.start.generate_velocities = true;
  else
    param.start.generate_velocities = false;
  
  switch(init){
    case 1:
      param.start.shake_pos=true;
      param.start.shake_vel=true;
      break;
    case 2:
      param.start.shake_pos=false;
      param.start.shake_vel=true;
      break;
    case 3:
      param.start.shake_pos=false;
      param.start.shake_vel=false;
      break;
    case 4:
      param.start.shake_pos=false;
      param.start.shake_vel=false;
      param.start.remove_com=false;
      break;
    default:
      io::messages.add("Illegal option for init in START block",
		       "In_Parameter", io::message::error); 
  }
  if(param.start.tempi <0)
    io::messages.add("Illegal value for TEMPI in START block (>=0)",
		     "In_Parameter", io::message::error);
  if(heat)
    io::messages.add("HEAT != 0 is not supported in START block",
		     "In_Parameter", io::message::error);
  if(ntx0!=1)
    io::messages.add("NTX0 != 1 is not supported in START block",
		     "In_Parameter", io::message::error);
  if(math::k_Boltzmann <=0)
    io::messages.add("BOLTZ <=0 is not appreciated in START block",
		     "In_Parameter", io::message::error);
}
/**
 * read CENTREOFMASS block.
 */
void io::In_Parameter::read_CENTREOFMASS(simulation::Parameter &param)
{
  DEBUG(8, "read CENTREOFMASS");

  std::vector<std::string> buffer;
  std::string s;
  
  DEBUG(10, "reading CENTREOFMASS block");
  buffer = m_block["CENTREOFMASS"];
  
  if (!buffer.size()){
    param.centreofmass.ndfmin = 0;
    param.start.remove_com=false;
    param.centreofmass.skip_step = 0;
    return;
  }

  block_read.insert("CENTREOFMASS");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  int ntcm;
  
  _lineStream >> param.centreofmass.ndfmin 
	      >> ntcm 
	      >> param.centreofmass.skip_step;
  
  if (_lineStream.fail())
    io::messages.add("bad line in CENTREOFMASS block",
		     "In_Parameter", io::message::error);
  if(ntcm!=0) 
    param.start.remove_com=true;
  
  if(param.centreofmass.ndfmin < 0)
    io::messages.add("Illegal value for NDFMIN in CENTREOFMASS block (>=0)",
		     "In_Parameter", io::message::error);
  if(param.centreofmass.skip_step <0)
    io::messages.add("Illegal value for NSCM in CENTREOFMASS block (>=0)",
		     "In_Parameter", io::message::error);
}

/**
 * read LONGRANGE block.
 */
void io::In_Parameter::read_LONGRANGE(simulation::Parameter &param)
{
  DEBUG(8, "read LONGRANGE");

  std::vector<std::string> buffer;
  std::string s;
  
  DEBUG(10, "longrange block");
  buffer = m_block["LONGRANGE"];
  
  if (!buffer.size()){
    io::messages.add("no LONGRANGE block in input",
		     "In_Parameter",io::message::error);
    param.longrange.rf_epsilon=1;
    param.longrange.rf_kappa=0;
    param.longrange.rf_cutoff=1;
    param.longrange.epsilon=1;
    return;
  }
  
  block_read.insert("LONGRANGE");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  _lineStream >> param.longrange.rf_epsilon 
	      >> param.longrange.rf_kappa 
	      >> param.longrange.rf_cutoff;
  param.longrange.epsilon = 1.0;
  
  if (_lineStream.fail())
    io::messages.add("bad line in LONGRANGE block",
		     "In_Parameter", io::message::error);
  
  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached, but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);
  */

  if(param.longrange.rf_cutoff < 0) {
    param.longrange.rf_excluded=true;
    param.longrange.rf_cutoff = -param.longrange.rf_cutoff;
    
    io::messages.add("Reaction field contribution of excluded atoms is " 
		     "taken into account", "In_Parameter", 
		     io::message::notice);
  }
  else{
    param.longrange.rf_excluded=false;
    io::messages.add("Reaction field contribution of excluded atoms is " 
		     "NOT taken into account", "In_Parameter", 
		     io::message::notice);
  }
  if(param.longrange.rf_epsilon!=0 && param.longrange.rf_epsilon<1)
    io::messages.add("Illegal value for EPSRF in LONGRANGE block (0  / >=1)", 
		     "In_Parameter", io::message::error);
  if(param.longrange.rf_kappa <0)
    io::messages.add("Illegal value for APPAK (who came up with this name?)"
		     " in LONGRANGE block (>=0)",
		     "In_Parameter", io::message::error);
} // LONGRANGE

/**
 * read SUBMOLECULES block.
 */
void io::In_Parameter::read_SUBMOLECULES(simulation::Parameter &param)
{
  DEBUG(8, "read SUBMOLECULES");

  std::vector<std::string> buffer;
  std::string s;
  
  DEBUG(10, "submolecules block");
  buffer = m_block["SUBMOLECULES"];
  
  if (!buffer.size()){
    io::messages.add("no SUBMOLECULES block in input",
		     "In_Parameter", io::message::error);
    return;
  }

  block_read.insert("SUBMOLECULES");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  int num;
  
  _lineStream >> num;

  if(num<0){
    io::messages.add("negative number of SUBMOLECULES is not allowed",
		     "In_Parameter", io::message::error);
    return;
  }
  
  size_t m;
  size_t old_m=0;
  
  for(int i=0; i<num; ++i){
    _lineStream >> m;
    param.submolecules.submolecules.push_back(m);
    DEBUG(10, "add submol " << m);
    if(m<old_m){
      io::messages.add("wrong order in SUBMOLECULES block",
		       "In_Parameter", io::message::error);
      return;
    }
    old_m=m;
  }
  
  if (_lineStream.fail())
    io::messages.add("bad line in SUBMOLECULES block",
		     "In_Parameter", io::message::error);
  
  
} // SUBMOLECULES


/**
 * read PLIST block.
 */
void io::In_Parameter::read_PLIST(simulation::Parameter &param)
{
  DEBUG(8, "read PLIST");

  std::vector<std::string> buffer;
  std::string s;

  DEBUG(10, "pairlist block");
  
  // try a PLIST03
  buffer = m_block["PLIST03"];
  if (buffer.size()){
    block_read.insert("PLIST03");

    std::string s1, s2, s3;
    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
    
    _lineStream >> s1 
		>> param.pairlist.skip_step 
		>> param.pairlist.cutoff_short
		>> param.pairlist.cutoff_long 
		>> s2
		>> s3;
    
    if (_lineStream.fail())
      throw std::runtime_error("bad line in PLIST03 block");
    
    std::transform(s1.begin(), s1.end(), s1.begin(), tolower);
    std::transform(s2.begin(), s2.end(), s2.begin(), tolower);
    std::transform(s3.begin(), s3.end(), s3.begin(), tolower);
    
    if (s1 == "grid") param.pairlist.grid = true;
    else if (s1 == "standard") param.pairlist.grid = false;
    else{
      io::messages.add("wrong pairlist algorithm chosen (allowed: standard, grid) in PLIST03 block",
		       "In_Parameter", io::message::error);
      param.pairlist.grid = false;
    }
    
    if (param.pairlist.grid){
      if (s2 == "auto") 
	param.pairlist.grid_size = 0.5 * param.pairlist.cutoff_short;
      else{
	param.pairlist.grid_size = atof(s2.c_str());
	if (errno){
	  io::messages.add("wrong pairlist grid size chosen (allowed: auto, [size]) in PLIST03 block",
			   "In_Parameter", io::message::error);
	  param.pairlist.grid_size = 0.5 * param.pairlist.cutoff_short;
	}
      }
    }
    else param.pairlist.grid_size = 0;

    if (s3 == "atomic") param.pairlist.atomic_cutoff = true;
    else if (s3 == "chargegroup") param.pairlist.atomic_cutoff = false;
    else {
      param.pairlist.atomic_cutoff = (atoi(s3.c_str()) != 0);
	if (errno){
	  io::messages.add("wrong cutoff type chosen (allowed: atomic, chargegroup)"
			   " in PLIST03 block",
			   "In_Parameter", io::message::error);
	  param.pairlist.atomic_cutoff = false;
	}
    }
  }
  else{
    buffer = m_block["PLIST"];
    
    if (!buffer.size()){
      io::messages.add("no PLIST block in input","In_Parameter",io::message::error);
      param.pairlist.grid_size = 0;
      param.pairlist.grid = false;
      param.pairlist.cutoff_short = 0;
      param.pairlist.cutoff_long = 0;
      param.pairlist.skip_step = 1;
      param.pairlist.atomic_cutoff = false;
      return;
    }

    block_read.insert("PLIST");

    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
    int i;
    
    _lineStream >> i 
		>> param.pairlist.skip_step  
		>> param.pairlist.cutoff_short 
		>> param.pairlist.cutoff_long;

    param.pairlist.grid_size = 0;
    param.pairlist.grid = false;
    param.pairlist.atomic_cutoff = false;

    if (_lineStream.fail()){
      io::messages.add("bad line in PLIST block",
		       "In_Parameter", io::message::error);
    }
    
    DEBUG(7, "pairlist update=" << param.pairlist.skip_step);
    DEBUG(7, "setting short cutoff=" << param.pairlist.cutoff_short 
	  << " long cutoff=" << param.pairlist.cutoff_long);
    
    
  }
  if(param.pairlist.grid && param.pairlist.grid_size <=0)
    io::messages.add("Illegal value for grid size in PLIST03 block (>0)",
		     "In_Parameter", io::message::error);
  if(param.pairlist.cutoff_short < 0){
    io::messages.add("Illegal value for short range cutoff in PLIST block (>0)",
		     "In_Parameter", io::message::error);
  }
  if(param.pairlist.cutoff_long < param.pairlist.cutoff_short){
    io::messages.add("Illegal value for long range cutoff in PLIST block (>=RCUTP)",
		     "In_Parameter", io::message::error);
  }
  
}


/**
 * read MULTIBATH block.
 */
void io::In_Parameter::read_MULTIBATH(simulation::Parameter &param)
{
  DEBUG(8, "read MULTIBATH");

  std::vector<std::string> buffer;
  std::string s;
  
  DEBUG(10, "TEMPERATURE COUPLING block");
  
  // is there a MULTIBATH block
  buffer = m_block["MULTIBATH"];
  
  if (buffer.size()){

    block_read.insert("MULTIBATH");

    param.multibath.found_multibath=true;
    param.multibath.found_tcouple=false;
    
    DEBUG(11, "MULTIBATH present");
    io::messages.add("using MULTIBATH block",
		     "In_Parameter", io::message::notice);

    _lineStream.clear();
    _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
    
    int num;
    size_t last;
    size_t com_bath, ir_bath;
    double temp, tau;
    
    // the baths
    _lineStream >> num;
    
    for(int i=0; i<num; ++i){
      _lineStream >> temp >> tau;
      
      param.multibath.multibath.add_bath(temp, tau);
    }
    
    if (_lineStream.fail()){
      io::messages.add("bad line in MULTIBATH block",
		       "In_Parameter", io::message::error);
    }
    
    // now the ranges
    _lineStream >> num;
    
    for(int i=0; i<num; ++i){
      _lineStream >> last >> com_bath >> ir_bath;
      param.multibath.multibath.add_bath_index(last - 1, 0, com_bath, ir_bath);
    }
    
    if (_lineStream.fail()){
      io::messages.add("bad line in MULTIBATH block",
		       "In_Parameter", io::message::error);
    }
    
  }
  else{
    // try a TCOUPLE block
    
    buffer = m_block["TCOUPLE"];
    if (buffer.size()){
      block_read.insert("TCOUPLE");

      param.multibath.found_multibath=false;
      param.multibath.found_tcouple=true;
      DEBUG(11, "TCOUPLE present");
      
      _lineStream.clear();
      _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));

      
      for(int i=0; i<3; ++i){
	_lineStream >> param.multibath.tcouple.ntt[i] 
		    >> param.multibath.tcouple.temp0[i]
		    >> param.multibath.tcouple.tau[i];
      }
      
      if (_lineStream.fail()){
	io::messages.add("bad line in TCOUPLE block",
			 "In_Parameter", io::message::error);
	return;
      }	
      
    }
    else{
      param.multibath.found_multibath=false;
      param.multibath.found_tcouple=false;
      DEBUG(11, "no TEMPERATURE COUPLING block");
      // no TCOUPLE block
      // that's fine, same as 0,0,0
    }
    
  }
  
} // TEMPERATURE coupling

/**
 * read POSREST block.
 */
void io::In_Parameter::read_POSREST(simulation::Parameter &param)
{
  DEBUG(8, "read POSREST");

  std::vector<std::string> buffer;
  std::string s;
  
  DEBUG(10, "posrest block");
  buffer = m_block["POSREST"];
  
  if (!buffer.size()){
    param.posrest.posrest=0;
    param.posrest.nrdrx = false;
    param.posrest.force_constant=1;
    return;
  }
  
  block_read.insert("POSREST");

  _lineStream.clear();
  _lineStream.str(concatenate(buffer.begin()+1, buffer.end()-1, s));
  
  _lineStream >> param.posrest.posrest
	      >> param.posrest.force_constant
	      >> param.posrest.nrdrx;
  
  if (_lineStream.fail())
    io::messages.add("bad line in POSREST block",
		     "In_Parameter", io::message::error);
  
  /*
  if (!_lineStream.eof())
    io::messages.add("End of line not reached, but should have been: \n" + s +  "\n",
		     "In_Parameter", io::message::warning);
  */

  if(param.posrest.posrest != 0) {
    io::messages.add("Position restraining / constraining currently not supported",
		     "In_Parameter", 
		     io::message::warning);
  }

  if(param.posrest.force_constant <0)
    io::messages.add("Illegal value for CHO"
		     " in POSREST block (>=0)",
		     "In_Parameter", io::message::error);

} // POSREST

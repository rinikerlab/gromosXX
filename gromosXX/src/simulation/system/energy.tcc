/**
 * @file energy.tcc
 * implements the energy methods.
 */

inline void simulation::Energy::zero()
{
  bond_energy.assign(bond_energy.size(), 0.0);
  angle_energy.assign(angle_energy.size(), 0.0);
  improper_energy.assign(improper_energy.size(), 0.0);
  dihedral_energy.assign(dihedral_energy.size(), 0.0);
  
  lj_energy.assign(lj_energy.size(), 
		   std::vector<double>(lj_energy.size(), 0.0));
  crf_energy.assign(crf_energy.size(), 
		    std::vector<double>(crf_energy.size(), 0.0));
}

inline void simulation::Energy::resize(size_t s)
{
  bond_energy.resize(s);
  angle_energy.resize(s);
  improper_energy.resize(s);
  dihedral_energy.resize(s);
  
  lj_energy.resize(s);
  crf_energy.resize(s);

  for(size_t i=0; i<s; ++i){
    lj_energy[i].resize(s);
    crf_energy[i].resize(s);  
  }

  zero();  
}

inline void simulation::Energy::calculate_totals()
{
  int num_groups = bond_energy.size();
  total = 0.0;
  kinetic_total = 0.0;
  potential_total = 0.0;
  bond_total = 0.0;
  angle_total = 0.0;
  improper_total = 0.0;
  dihedral_total = 0.0;
  nonbonded_total = 0.0;
  lj_total = 0.0;
  crf_total = 0.0;
  special_total = 0.0;
    
  for(std::vector<double>::const_iterator it = kinetic_energy.begin(),
	to = kinetic_energy.end(); it != to; ++it)
    kinetic_total += *it;
   
  for(int i=0; i<num_groups; i++){
    for(int j=0; j<num_groups; j++){
      lj_total   += lj_energy[i][j];
      crf_total  += crf_energy[i][j];
    }

    bond_total     += bond_energy[i];
    angle_total    += angle_energy[i];
    improper_total += improper_energy[i];
    dihedral_total += dihedral_energy[i];
    // tot_posrest += posrest_energy[i];
      
  }
  nonbonded_total = lj_total + crf_total;
  bonded_total    = bond_total + angle_total + 
    dihedral_total + improper_total;
  potential_total = nonbonded_total + bonded_total;
  // special_total   = posrest_total;

  total = potential_total + kinetic_total + special_total;

}

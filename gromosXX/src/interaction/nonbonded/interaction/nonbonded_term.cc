/**
 * @file nonbonded_term.cc
 * inline methods of Nonbonded_Term
 */

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE nonbonded

#include <interaction/nonbonded/interaction/latticesum.h>

/**
 * helper function to initialize the constants.
 */
inline void interaction::Nonbonded_Term
::init(simulation::Simulation const &sim)
{
  switch(sim.param().force.interaction_function){
  case simulation::lj_crf_func :
  case simulation::pol_lj_crf_func :
    // Force
    m_cut3i = 
      1.0 / ( sim.param().nonbonded.rf_cutoff
	      * sim.param().nonbonded.rf_cutoff
	      * sim.param().nonbonded.rf_cutoff);
    DEBUG(15, "nonbonded term init: m_cut3i: " << m_cut3i);
    m_crf = 2*(sim.param().nonbonded.epsilon - sim.param().nonbonded.rf_epsilon) * 
      (1.0 + sim.param().nonbonded.rf_kappa * sim.param().nonbonded.rf_cutoff) -
      sim.param().nonbonded.rf_epsilon * (sim.param().nonbonded.rf_kappa  * 
					  sim.param().nonbonded.rf_cutoff *
					  sim.param().nonbonded.rf_kappa  *
					  sim.param().nonbonded.rf_cutoff);
    
    m_crf /= (sim.param().nonbonded.epsilon +2* sim.param().nonbonded.rf_epsilon) *
      (1.0 + sim.param().nonbonded.rf_kappa * sim.param().nonbonded.rf_cutoff) +
      sim.param().nonbonded.rf_epsilon * (sim.param().nonbonded.rf_kappa  * 
					  sim.param().nonbonded.rf_cutoff *
					  sim.param().nonbonded.rf_kappa  *
					  sim.param().nonbonded.rf_cutoff);
    DEBUG(15, "nonbonded term init: m_crf: " << m_crf);
    m_crf_cut3i = m_crf * m_cut3i;
    
    // Energy
    m_crf_2cut3i = m_crf_cut3i / 2.0;
    DEBUG(15, "nonbonded term init: m_crf_2cut3i: " << m_crf_2cut3i);
    
    m_crf_cut = (1 - m_crf / 2.0)
      / sim.param().nonbonded.rf_cutoff;
    break;
    
  case simulation::cgrain_func :
    // cgrain
    A_cg12= - (12.0 * (12 + 4)) / (pow(sim.param().nonbonded.rf_cutoff, 12 + 3));
    A_cg6=  - (6.0  * (6  + 4)) / (pow(sim.param().nonbonded.rf_cutoff, 6  + 3));
    A_cg1=  - (1.0  * (1  + 4)) / (pow(sim.param().nonbonded.rf_cutoff, 1  + 3));     
    
    B_cg12=   (12.0 * (12 + 3)) / (pow(sim.param().nonbonded.rf_cutoff, 12 + 4));
    B_cg6=    (6.0  * (6  + 3)) / (pow(sim.param().nonbonded.rf_cutoff, 6  + 4));
    B_cg1=    (1.0  * (1  + 3)) / (pow(sim.param().nonbonded.rf_cutoff, 1  + 4));     

    C_cg12=   ((12 + 3) * (12 + 4)) / (12.0 * pow(sim.param().nonbonded.rf_cutoff, 12));
    C_cg6=    ((6  + 3) * (6  + 4)) / (12.0 * pow(sim.param().nonbonded.rf_cutoff, 6 ));
    C_cg1=    ((1  + 3) * (1  + 4)) / (12.0 * pow(sim.param().nonbonded.rf_cutoff, 1 ));
    
    cgrain_eps = sim.param().cgrain.EPS;
    break;
  case simulation::lj_ls_func :
    // lattice sum
    charge_shape = sim.param().nonbonded.ls_charge_shape;
    charge_width_i = 1.0 / sim.param().nonbonded.ls_charge_shape_width;
    break;
  default:
    io::messages.add("Nonbonded_Innerloop",
		     "interaction function not implemented",
		     io::message::critical);
  }
  m_cut2 = sim.param().pairlist.cutoff_long * sim.param().pairlist.cutoff_long;
}

/**
 * helper function to calculate the force and energy for
 * a given atom pair.
 */
inline void interaction::Nonbonded_Term
::lj_crf_interaction(math::Vec const &r,
		     double c6, double c12,
		     double q,
		     double &force, double &e_lj, double &e_crf)
{
  DEBUG(14, "\t\tnonbonded term");
  
  assert(abs2(r) != 0);
  const double dist2 = abs2(r);
  const double dist2i = 1.0 / dist2;
  const double q_eps = q * math::four_pi_eps_i;
  const double dist6i = dist2i * dist2i * dist2i;
  const double disti = sqrt(dist2i);
  const double c12_dist6i = c12 * dist6i;
  
  e_lj = (c12_dist6i - c6) * dist6i;

  e_crf = q_eps * 
      (disti - m_crf_2cut3i * dist2 - m_crf_cut);

  force = (c12_dist6i + c12_dist6i - c6) * 6.0 * dist6i * dist2i + 
      q_eps * (disti * dist2i + m_crf_cut3i);

  DEBUG(15, "\t\tq=" << q << " 4pie=" << math::four_pi_eps_i 
	<< " crf_cut2i=" << m_crf_cut3i);
  
}

/**
 * helper function to calculate the force and energy for
 * a given atom pair.
 */
inline void interaction::Nonbonded_Term
::lj_interaction(math::Vec const &r,
		     double c6, double c12,
		     double &force, double &e_lj)
{
  DEBUG(14, "\t\tnonbonded term");
  
  assert(abs2(r) != 0);
  const double dist2 = abs2(r);
  const double dist2i = 1.0 / dist2;
  const double dist6i = dist2i * dist2i * dist2i;
  const double c12_dist6i = c12 * dist6i;
  
  e_lj = (c12_dist6i - c6) * dist6i;

  force = (c12_dist6i + c12_dist6i - c6) * 6.0 * dist6i * dist2i;
  
}

inline void interaction::Nonbonded_Term
::lj_ls_interaction(math::Vec const &r, double c6, double c12, double q,
		 double &force, double &e_lj, double &e_ls)
{
  DEBUG(14, "\t\tnonbonded term");
  
  assert(abs2(r) != 0);
  const double dist2 = abs2(r);
  const double dist = sqrt(dist2);
  const double dist2i = 1.0 / dist2;
  const double dist6i = dist2i * dist2i * dist2i;
  const double q_eps = q * math::four_pi_eps_i;
  const double disti = 1.0 / dist;
  const double c12_dist6i = c12 * dist6i;
  
  const double ai_dist = charge_width_i * dist;
  
  e_lj = (c12_dist6i - c6) * dist6i;
  double eta;
  double d_eta;
  interaction::Lattice_Sum::charge_shape_switch(charge_shape, ai_dist, eta, d_eta);
  DEBUG(14, "eta: " << eta << " d_eta: " << d_eta);
  e_ls = q_eps * disti * eta;
  //         qi * qj       eta      eta'
  // force = ------- * (-  ---  +  ---- )
  //          r_ij^2       r_ij      a
  const double f_ls = dist2i * (e_ls - q_eps * d_eta * charge_width_i);
  force = (c12_dist6i + c12_dist6i - c6) * 6.0 * dist6i * dist2i + f_ls;
          
  DEBUG(14, "e_ls: " << e_ls << ", f_ls: " << f_ls);
}

inline void interaction::Nonbonded_Term
::ls_excluded_interaction(math::Vec const &r, double q,
		 double &force, double &e_ls)
{
  DEBUG(14, "\t\tnonbonded term");
  
  assert(abs2(r) != 0);
  const double dist2 = abs2(r);
  const double dist = sqrt(dist2);
  const double dist2i = 1.0 / dist2;
  const double q_eps = q * math::four_pi_eps_i;
  const double disti = 1.0 / dist;
  const double ai_dist = charge_width_i * dist;

  double eta;
  double d_eta;
  interaction::Lattice_Sum::charge_shape_switch(charge_shape, ai_dist, eta, d_eta);
  DEBUG(14, "eta: " << eta << " d_eta: " << d_eta);
  e_ls = q_eps * disti * (eta - 1.0);
  force = dist2i * (e_ls - q_eps * d_eta * charge_width_i);
}

/**
 * helper function to calculate the force and energy for
 * a given atom pair (polarisable).
 */
inline void interaction::Nonbonded_Term
::pol_lj_crf_interaction(math::Vec const &r,
                     math::Vec const &rp1,
                     math::Vec const &rp2,
                     math::Vec const &rpp,
		     double c6, double c12,
		     double qi, double qj, double cgi, double cgj,
		     double f[], double &e_lj, double &e_crf)
{
  DEBUG(14, "\t\tnonbonded term");
  
  assert(abs2(r) != 0);

  const double dist2 = abs2(r);
  const double dist2p1 = abs2(rp1);
  const double dist2p2 = abs2(rp2);
  const double dist2pp = abs2(rpp);
  const double dist2i = 1.0 / dist2;
  const double dist2p1i = 1.0 / dist2p1;
  const double dist2p2i = 1.0 / dist2p2;
  const double dist2ppi = 1.0 / dist2pp;

  const double dist6i = dist2i * dist2i * dist2i;

  const double disti = sqrt(dist2i);
  const double distp1i = sqrt(dist2p1i);
  const double distp2i = sqrt(dist2p2i);
  const double distppi = sqrt(dist2ppi);

  const double c12_dist6i = c12 * dist6i;
  const double qi_m_cgi = qi - cgi;
  const double qj_m_cgj = qj - cgj;
  const double q_eps = (qi_m_cgi)*(qj_m_cgj) * math::four_pi_eps_i;
  const double q_epsp1 = (qi_m_cgi)*cgj * math::four_pi_eps_i;
  const double q_epsp2 = cgi*(qj_m_cgj) * math::four_pi_eps_i;
  const double q_epspp = cgi*cgj * math::four_pi_eps_i;
  
  e_lj = (c12_dist6i - c6) * dist6i;

  e_crf = q_eps * (disti - m_crf_2cut3i * dist2 - m_crf_cut)
    + q_epsp1 * (distp1i - m_crf_2cut3i * dist2p1 - m_crf_cut)
    + q_epsp2 * (distp2i - m_crf_2cut3i * dist2p2 - m_crf_cut)
    + q_epspp * (distppi - m_crf_2cut3i * dist2pp - m_crf_cut);

  f[0] = (c12_dist6i + c12_dist6i - c6) * 6.0 * dist6i * dist2i + 
         q_eps * (disti * dist2i + m_crf_cut3i);
  f[1] = q_epsp1 * (distp1i * dist2p1i + m_crf_cut3i);
  f[2] = q_epsp2 * (distp2i * dist2p2i + m_crf_cut3i);
  f[3] = q_epspp * (distppi * dist2ppi + m_crf_cut3i);

  DEBUG(15, "\t\tq=" << qi*qj << " 4pie=" << math::four_pi_eps_i 
	<< " crf_cut2i=" << m_crf_cut3i);
  
}

/**
 * helper function to calculate the force and energy for
 * the reaction field contribution for a given pair
 */
inline void interaction::Nonbonded_Term
::rf_interaction(math::Vec const &r,double q,
		 math::Vec &force, double &e_crf)
{
  const double dist2 = abs2(r);
  
  force = q * math::four_pi_eps_i *  m_crf_cut3i * r;

  e_crf = q * math::four_pi_eps_i * ( -m_crf_2cut3i * dist2 - m_crf_cut);
  DEBUG(11, "dist2 " << dist2 );
  DEBUG(11, "crf_2cut3i " << m_crf_2cut3i);
  DEBUG(11, "crf_cut " << m_crf_cut);
  DEBUG(11, "q*q   " << q );
  
}

/**
 * helper function to calculate the force and energy for
 * the reaction field contribution for a given pair
 * with polarisation
 */
inline void interaction::Nonbonded_Term
::pol_rf_interaction(math::Vec const &r,
                 math::Vec const &rp1,
                 math::Vec const &rp2,
                 math::Vec const &rpp,
                 double qi, double qj, 
                 double cgi, double cgj,
		 math::VArray &force, double &e_crf)
{
  const double dist2 = abs2(r);
  const double dist2p1 = abs2(rp1);
  const double dist2p2 = abs2(rp2);
  const double dist2pp = abs2(rpp);

  const double eps = math::four_pi_eps_i;
  const double qeps = (qi-cgi)*(qj-cgj) * eps;
  const double qepsp1 = (qi-cgi)*cgj * eps;
  const double qepsp2 = cgi*(qj-cgj) * eps;
  const double qepspp = cgi*cgj * eps;
  
  force(0) = qeps *  m_crf_cut3i * r;
  force(1) = qepsp1 *  m_crf_cut3i * rp1;
  force(2) = qepsp2 *  m_crf_cut3i * rp2;
  force(3) = qepspp *  m_crf_cut3i * rpp;

  e_crf = qeps*( -m_crf_2cut3i*dist2 - m_crf_cut)
          + qepsp1*( -m_crf_2cut3i*dist2p1 - m_crf_cut)
          + qepsp2*( -m_crf_2cut3i*dist2p2 - m_crf_cut)
          + qepspp*( -m_crf_2cut3i*dist2pp - m_crf_cut);
}


/**
 * helper function to calculate the force and energy for
 * a given atom pair in the coarse grain model
 */
inline void interaction::Nonbonded_Term
::cgrain_interaction(math::Vec const &r,
                     double c6, double c12,
                     double q,
                     double &force, double &e_lj, double &e_crf)
{
  assert(abs2(r) != 0);
  const double dist2 = abs2(r);
  const double dist2i = 1.0 / dist2;
  const double disti = sqrt(dist2i);
  const double dist6i = dist2i * dist2i * dist2i;
  const double dist12i = dist6i * dist6i;
  const double dist = 1.0 / disti;

  const double q_eps = q * math::four_pi_eps_i / cgrain_eps;

  // const double c12_dist6i = c12 * dist6i;

  e_crf = (q_eps * (disti
                    - A_cg1  / 3 * dist2 * dist
                    - B_cg1  / 4 * dist2 * dist2
                    - C_cg1 ));
  
  e_lj =  (c12 * (dist12i
		  - A_cg12 / 3 * dist2 * dist
		  - B_cg12 / 4 * dist2 * dist2
		  - C_cg12))
    -     (c6 *  (dist6i
		  - A_cg6  / 3 * dist2 * dist
		  - B_cg6  / 4 * dist2 * dist2 
		  - C_cg6 ));
  
  force = c12 * (12.0 * dist12i * disti + A_cg12 * dist2 + B_cg12 * dist2 * dist) * disti -
           c6 * ( 6.0 * dist6i *  disti + A_cg6  * dist2 + B_cg6  * dist2 * dist) * disti +
               q_eps * (         dist2i + A_cg1  * dist2 + B_cg1  * dist2 * dist) * disti;
  
  
  std::cout.precision(10);
  
  DEBUG(11, "r_ij= " << dist 
        << " e_lj=" << e_lj << " e_crf=" << e_crf 
        << " force=" << force);
  
}

inline double interaction::Nonbonded_Term
::crf_2cut3i()const
{
  return m_crf_2cut3i;
}

/**
 * helper function to calculate a term of the electric field 
 * at a given position for the polarisation
 */
inline void interaction::Nonbonded_Term
::electric_field_interaction(math::Vec const &r, 
                       math::Vec const &rprime, 
                       double qj, double charge, 
                       math::Vec &e_el) {

  DEBUG(14, "\t\tenergy field term for polarisation");

  assert(abs2(r) != 0);
  assert(abs2(rprime) != 0);
  const double distj = abs2(r);
  const double distp = abs2(rprime);
  const double distji = 1.0 / (distj*sqrt(distj));
  const double distpi = 1.0 / (distp*sqrt(distp));
  const double q_eps = (qj-charge) * math::four_pi_eps_i;
  const double q_epsp = charge * math::four_pi_eps_i;

  e_el = (q_eps*(distji + m_crf_cut3i))*r + (q_epsp*(distpi + m_crf_cut3i))*rprime;
}

/**
 * helper function to calculate the self energy 
 * at a given atom.
 */
inline void interaction::Nonbonded_Term
::self_energy_interaction(double alpha, double e_i2, double &self_e) {

  DEBUG(14, "\t\tself energy - dipole-dipole interaction");
  self_e = 0.5 * alpha * e_i2;
}

/**
 * helper function to calculate the self energy 
 * at a given atom (damped).
 */
inline void interaction::Nonbonded_Term
::self_energy_interaction(double alpha, double e_i2, double e_0, double p,
        double &self_e) {

    DEBUG(14, "\t\tself energy - dipole-dipole interaction");
    const double e_02 = e_0 * e_0;
    if (e_i2 <= e_02) {
        self_e = 0.5 * alpha * e_i2;
    } else {
        const double e_i = sqrt(e_i2);
        self_e = 0.5 * alpha * e_02 +
                0.5 * alpha * e_02 / (p * (p - 1)) *
                (-p * p +
                (e_i / e_0)*(p * p - 1) +
                pow(e_0 / e_i, p - 1));
    }
}

inline void
interaction::Nonbonded_Term::lj_crf_hessian(math::Vec const &r,
				    double c6, double c12,
				    double q,
				    math::Matrix &hess)
{
  const double r2 = math::abs2(r);
  
  const double r4 = r2*r2;
  const double r8 = r4*r4;
  const double r10 = r8*r2;
  const double r14 = r10*r4;
  const double r16 = r8*r8;
    
  // the LENNARD-JONES part
  
  // get the matrix for the first term
  math::dyade(r, r, hess);

  for(int d1=0; d1 < 3; ++d1){
    // first term
    for(int d2=0; d2 < 3; ++d2){
      hess(d1, d2) *= 168.0 * c12 / r16 - 48.0 * c6 / r10;
    }
    // second term
    hess(d1, d1) += 6.0 * c6 / r8 - 12.0 * c12 / r14;
  }

  const double r3 = sqrt(r4 * r2);
  math::Matrix c;
  math::dyade(r, r, c);

  for(int d1=0; d1<3; ++d1){
    for(int d2=0; d2<3; ++d2){
      c(d1, d2) *= 3.0 / r2;
    }
    c(d1, d1) -= 1.0;
  }

  for(int d1=0; d1<3; ++d1){
    for(int d2=0; d2<3; ++d2){
      // first factor
      c(d1, d2) *= q * math::four_pi_eps_i / r3;
    }
    // reaction field term
    c(d1, d1) -= q * math::four_pi_eps_i * m_crf_cut3i;
  }

  for(int d1=0; d1<3; ++d1){
    for(int d2=0; d2<3; ++d2){
      // first factor
      hess(d1, d2) += c(d1, d2);
    }
  }
  
}

/**
 * calculate the force and energy of an atom pair (sasa).
 */
inline void interaction::Nonbonded_Term
::sasa_interaction(math::Vec const &r, double bij,
                   double pij, double p_i, double surface,
                   double & e_sasa)
{
  DEBUG(14, "\t\tsasa term");

  assert(math::abs(r) != 0);
  assert(surface != 0);

  e_sasa = (1 - (p_i * pij * bij / surface));

}


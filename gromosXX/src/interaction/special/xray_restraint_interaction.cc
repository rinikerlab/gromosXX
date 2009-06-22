/**
 * @file xray_restraint_interaction.cc
 * template methods of Xray_Restraint_Interaction
 */

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>
#include <interaction/interaction.h>

#include <math/periodicity.h>

// special interactions
#include <interaction/interaction_types.h>

#include <interaction/special/xray_restraint_interaction.h>

#include <util/template_split.h>
#include <util/debug.h>
#include <vector>
#include <string>
#include <ios>


#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special
//#define HAVE_CLIPPER
interaction::Xray_Restraint_Interaction::Xray_Restraint_Interaction() : Interaction("XrayRestraint") {
}

interaction::Xray_Restraint_Interaction::~Xray_Restraint_Interaction() {
}

/**
 * calculate xray restraint interactions
 */
template<math::boundary_enum B, math::virial_enum V>
void interaction::Xray_Restraint_Interaction::_calculate_xray_restraint_interactions
(topology::Topology & topo,
        configuration::Configuration & conf,
        simulation::Simulation & sim,
        int & error) {
#ifdef HAVE_CLIPPER
  m_timer.start();
  error = 0;
  // get number of atoms in simulation
  const unsigned int atoms_size = topo.num_atoms();
  //update clipper atomvec
  for (unsigned int i = 0; i < atoms_size; i++) {
    atoms[i].set_coord_orth(clipper::Coord_orth(conf.current().pos(i)(0)*10.0,
            conf.current().pos(i)(1)*10.0,
            conf.current().pos(i)(2)*10.0));
  }
  // Calculate structure factors
  clipper::SFcalc_iso_fft<double> sfc;
  // run it
  sfc(fphi, atoms);

  // sqr_calc:       sum of squared Fcalc
  // obs:            sum of Fobs
  // calc:           sum of Fcalc
  // obs_calc:       sum of Fobs*Fcalc
  // obs_calcavg:    sum of Fobs*Fcalc(averaged)
  // obs_k_calcavg:  sum of Fobs-k_avg*Fcalc(averaged)
  // obs_k_calc:     sum of Fobs-k*Fcalc
  // sqr_calcavg:    sum of squared time-averaged Fcalc
  // calcavg:        sum of time-averaged Fcalc
  double sqr_calc = 0.0, obs = 0.0, calc = 0.0, obs_calc = 0.0, obs_k_calc = 0.0,
          sqr_calcavg = 0.0, calcavg = 0.0, obs_calcavg = 0.0, obs_k_calcavg = 0.0;
  // Number of reflections
  const unsigned int num_xray_rest = topo.xray_restraints().size();
  // e-term for time-average
  const double eterm = exp(-sim.time_step_size() / sim.param().xrayrest.tau);

  for (unsigned int i = 0; i < num_xray_rest; i++) {
    //filter calculated sf's
    clipper::HKL hkl(topo.xray_restraints()[i].h, topo.xray_restraints()[i].k, topo.xray_restraints()[i].l);
    conf.special().xray_rest[i].sf_curr = fabs(fphi[hkl].f());
    conf.special().xray_rest[i].phase_curr = fphi[hkl].phi();
    DEBUG(15,"HKL:" << hkl.h() << "," << hkl.k() << "," << hkl.l()); 
    DEBUG(15,"\tSF: " << conf.special().xray_rest[i].sf_curr);

    if (!sim.param().xrayrest.readavg && sim.steps() == 0) {
      // reset the averages at the beginning if requested
      conf.special().xray_rest[i].sf_av = conf.special().xray_rest[i].sf_curr;
      conf.special().xray_rest[i].phase_av = conf.special().xray_rest[i].phase_curr;
    }
    conf.special().xray_rest[i].sf_av = fabs((1.0 - eterm) * conf.special().xray_rest[i].sf_curr + eterm * conf.special().xray_rest[i].sf_av);
    conf.special().xray_rest[i].phase_av = (1.0 - eterm) * conf.special().xray_rest[i].phase_curr + eterm * conf.special().xray_rest[i].phase_av;

    // calc sums
    obs_calc += conf.special().xray_rest[i].sf_curr * topo.xray_restraints()[i].sf;
    obs_calcavg += conf.special().xray_rest[i].sf_av * topo.xray_restraints()[i].sf;
    sqr_calc += conf.special().xray_rest[i].sf_curr * conf.special().xray_rest[i].sf_curr;
    obs += topo.xray_restraints()[i].sf;
    calc += conf.special().xray_rest[i].sf_curr;
    sqr_calcavg += conf.special().xray_rest[i].sf_av * conf.special().xray_rest[i].sf_av;
    calcavg += conf.special().xray_rest[i].sf_av;
  }

  // check for possible resolution problems
#ifdef HAVE_ISNAN
  if (std::isnan(calc)){
    io::messages.add("Structure factors were NaN. This can be due to numerical problems. "
                     "Try to slighlty increase the resolution.", "X-Ray Restraints", io::message::error);
    error = 1;
  }
#endif



  //calc k_inst and k_avg
  double & k_inst = conf.special().xray.k_inst;
  k_inst = obs_calc / sqr_calc;
  double & k_avg = conf.special().xray.k_avg;
  k_avg = obs_calcavg / sqr_calcavg;
  DEBUG(10, "k_inst value: " << k_inst);
  DEBUG(10, "k_avg  value: " << k_avg);

  for (unsigned int i = 0; i < num_xray_rest; i++) {
    obs_k_calc += fabs(topo.xray_restraints()[i].sf - k_inst * conf.special().xray_rest[i].sf_curr);
    obs_k_calcavg += fabs(topo.xray_restraints()[i].sf - k_avg * conf.special().xray_rest[i].sf_av);
  }

  // calculate R_inst and R_avg
  double & R_inst = conf.special().xray.R_inst;
  R_inst = obs_k_calc / obs;
  double & R_avg = conf.special().xray.R_avg;
  R_avg = obs_k_calcavg / obs;
  DEBUG(10, "R_inst value: " << std::setw(15) << std::setprecision(8) << R_inst);
  DEBUG(10, "R_avg  value: " << std::setw(15) << std::setprecision(8) << R_avg);

  // calculate gradient
  D_k = std::complex<double> (0.0f, 0.0f); // zero it

  double energy_sum = 0.0;
  for (unsigned int i = 0; i < num_xray_rest; i++) {
    const topology::xray_restraint_struct & xrs = topo.xray_restraints()[i];
    clipper::HKL hkl(xrs.h, xrs.k, xrs.l);
    // SWITCH FOR DIFFERENT METHODS
    switch (sim.param().xrayrest.xrayrest) {
      case simulation::xrayrest_off :
      {
        break;
      }
      case simulation::xrayrest_inst :
      {
        // INSTANTANEOUS
        // calculate energy-sum
        const double fobs = xrs.sf;
        const double fcalc = conf.special().xray_rest[i].sf_curr;
        const double term = fobs - k_inst * fcalc;
        energy_sum += term * term;
        // calculate derivatives of target function
        const double dterm = (k_inst * fcalc - fobs) * k_inst;
        D_k.set_data(hkl, clipper::data64::F_phi(sim.param().xrayrest.force_constant * dterm, conf.special().xray_rest[i].phase_curr));

        fphi_print.set_data(hkl, clipper::data64::F_phi(fobs/k_inst, conf.special().xray_rest[i].phase_curr));
        break;
      }
      case simulation::xrayrest_avg :
      {
        // TIMEAVERAGED
        // calculate energy-sum
        const double fobs = xrs.sf;
        const double fcalc = conf.special().xray_rest[i].sf_av;
        const double term = fobs - k_avg * fcalc;
        energy_sum += term * term;
        // calculate derivatives of target function
        // here we omit the 1-exp(-dt/tau) term.
        const double dterm = (k_avg * fcalc - fobs) * k_avg;
        D_k.set_data(hkl, clipper::data64::F_phi(sim.param().xrayrest.force_constant * dterm, conf.special().xray_rest[i].phase_curr));

        fphi_print.set_data(hkl, clipper::data64::F_phi(fobs/k_avg, conf.special().xray_rest[i].phase_av));
        break;
      }
      case simulation::xrayrest_biq :
      {
        // BIQUADRATIC TIME-AVERAGED/INSTANTANEOUS
        // calculate energy-sum
        const double fobs = xrs.sf;
        const double finst = conf.special().xray_rest[i].sf_curr;
        const double favg = conf.special().xray_rest[i].sf_av;
        const double inst_term = fobs - k_inst * finst;
        const double av_term = fobs - k_avg * favg;
        energy_sum += (inst_term * inst_term)*(av_term * av_term);
        // calculate derivatives of target function
        // here we omit the 1-exp(-dt/tau) term.
        const double dterm = (k_inst * finst - fobs)*(av_term * av_term) * k_inst
                + (k_avg * favg - fobs)*(inst_term * inst_term) * k_avg;
        D_k.set_data(hkl, clipper::data64::F_phi(sim.param().xrayrest.force_constant * dterm, conf.special().xray_rest[i].phase_curr));
        
        fphi_print.set_data(hkl, clipper::data64::F_phi(xrs.sf/k_avg, conf.special().xray_rest[i].phase_av));
        break;
      }
      case simulation::xrayrest_loel :
      {
        break;
      }
    }
  }
  // finally calc energy
  conf.current().energies.xray_total = 0.5 * sim.param().xrayrest.force_constant * energy_sum;
  DEBUG(10, "energy: " << conf.current().energies.xray_total);

  // perform FFT
  d_r.fft_from(D_k);
  clipper::Coord_grid g0, g1;

  // 3.5 is hardcoded atom radius for grid sampling.
  const double radius = 3.0;
  clipper::Grid_range gd(fphi.base_cell(), d_r.grid_sampling(), radius);

  clipper::Xmap<clipper::ftype64>::Map_reference_coord i0, iu, iv, iw;
  const double volume = fphi.base_cell().volume();
  
  // calculate gradients of structure factors
  for (unsigned int i = 0; i < atoms_size; i++) {
    if (!atoms[i].is_null()) {
      math::Vec gradient(0.0, 0.0, 0.0);
      clipper::AtomShapeFn sf(atoms[i].coord_orth(), atoms[i].element(),
              atoms[i].u_iso(), atoms[i].occupancy());
      sf.agarwal_params().resize(3);
      // conversion for clipper coordinate enum
      sf.agarwal_params()[0] = clipper::AtomShapeFn::X;
      sf.agarwal_params()[1] = clipper::AtomShapeFn::Y;
      sf.agarwal_params()[2] = clipper::AtomShapeFn::Z;
      // determine grid-ranges
      clipper::Coord_frac uvw = atoms[i].coord_orth().coord_frac(fphi.base_cell());
      g0 = uvw.coord_grid(d_r.grid_sampling()) + gd.min();
      g1 = uvw.coord_grid(d_r.grid_sampling()) + gd.max();
      i0 = clipper::Xmap<clipper::ftype64>::Map_reference_coord(d_r, g0);
      std::vector<clipper::ftype> rho_grad(3, 0.0f);
      clipper::ftype temp_rho = 0.0f;
      // loop over grid and convolve with the atomic density gradient
      unsigned int points = 0;
      for (iu = i0; iu.coord().u() <= g1.u(); iu.next_u()) {
        for (iv = iu; iv.coord().v() <= g1.v(); iv.next_v()) {
          for (iw = iv; iw.coord().w() <= g1.w(); iw.next_w(), ++points) {
            // get gradient from clipper
            sf.rho_grad(iw.coord_orth(), temp_rho, rho_grad);
            const double d_r_iw = d_r[iw];
            gradient(0) += d_r_iw * rho_grad[0];
            gradient(1) += d_r_iw * rho_grad[1];
            gradient(2) += d_r_iw * rho_grad[2];
          }
        }
      } // loop over map
      // convert from Angstrom to nm and add very annyoing scaling constants
      // to make the force volume AND resolution independent.
      gradient *= 10.0 / 2.0 * volume * volume / (d_r.grid_sampling().size());
      // add to force
      conf.current().force(i) -= gradient;
      DEBUG(10, "grad(" << i << "): " << math::v2s(gradient));
    } // if atom not null
  } // for atoms

  m_timer.stop();

  // write xmap to external file
  if (sim.param().xrayrest.writexmap != 0 && sim.steps() % sim.param().xrayrest.writexmap == 0) {
    const clipper::Grid_sampling grid(fphi.base_hkl_info().spacegroup(), fphi.base_cell(), fphi.base_hkl_info().resolution(), 1.5);
    clipper::Xmap<clipper::ftype64> density(fphi.base_hkl_info().spacegroup(), fphi.base_cell(), grid);
    density.fft_from(fphi_print);
    clipper::CCP4MAPfile mapfile;
    std::ostringstream file_name, asu_file_name;
    file_name << "density_frame_" << std::setw(int(log10(sim.param().step.number_of_steps)))
            << std::setfill('0') << sim.steps() << ".ccp4";
    if (sim.param().xrayrest.writedensity == 1 || sim.param().xrayrest.writedensity == 3) {
      mapfile.open_write(file_name.str());
      mapfile.export_xmap(density);
      mapfile.close_write();
    }
    // Non Cristallographic Map
    clipper::NXmap<double> asu(density.grid_asu(), density.operator_orth_grid());
    asu_file_name << "density_asu_frame_" << std::setw(int(log10(sim.param().step.number_of_steps)))
            << std::setfill('0') << sim.steps() << ".ccp4";
    if (sim.param().xrayrest.writedensity == 2 || sim.param().xrayrest.writedensity == 3) {
      mapfile.open_write(asu_file_name.str());
      mapfile.export_nxmap(asu);
      mapfile.close_write();
    }
  }
#endif
}

int interaction::Xray_Restraint_Interaction
::calculate_interactions(topology::Topology &topo,
        configuration::Configuration &conf,
        simulation::Simulation &sim) {
  int error;
  SPLIT_VIRIAL_BOUNDARY(_calculate_xray_restraint_interactions,
          topo, conf, sim, error);

  return error;
}

int interaction::Xray_Restraint_Interaction::init(topology::Topology &topo,
        configuration::Configuration &conf,
        simulation::Simulation &sim,
        std::ostream &os,
        bool quiet) {
#ifdef HAVE_CLIPPER
  DEBUG(15, "Xray_Restraint_Interaction: init")
  const double sqpi2 = (math::Pi * math::Pi * 8.0);

  // Redirect clipper errors
  clipper::Message message;
  message.set_stream(os);

  // Construct clipper objects
  clipper::Spacegroup spacegr;
  try {
    clipper::Spgr_descr spgrinit(clipper::String(sim.param().xrayrest.spacegroup), clipper::Spgr_descr::HM);
    spacegr.init(spgrinit);
  } catch (const clipper::Message_fatal & msg) {
    io::messages.add("Xray_restraint_interaction", msg.text(), io::message::error);
    return 1;
  }

  math::Box & box = conf.current().box;
  const double a = math::abs(box(0));
  const double b = math::abs(box(1));
  const double c = math::abs(box(2));
  const double alpha = acos(math::costest(dot(box(1), box(2)) / (b * c)));
  const double beta = acos(math::costest(dot(box(0), box(2)) / (a * c)));
  const double gamma = acos(math::costest(dot(box(0), box(1)) / (a * b)));
  clipper::Cell_descr cellinit(a * 10.0, b * 10.0, c * 10.0,
          alpha, beta, gamma);
  clipper::Cell cell(cellinit);

  clipper::Resolution reso(sim.param().xrayrest.resolution * 10.0);

  hkls.init(spacegr, cell, reso, true);
  fphi.init(hkls, hkls.cell());
  fphi_print.init(hkls, hkls.cell());
  D_k.init(hkls, hkls.cell());
  // 1.5 is shannon-rate for oversampled FFT
  const clipper::Grid_sampling grid(fphi.base_hkl_info().spacegroup(), fphi.base_cell(), fphi.base_hkl_info().resolution(), 1.5);
  d_r.init(fphi.base_hkl_info().spacegroup(), fphi.base_cell(), grid);

  // Fill clipper atom-vector
  std::vector<clipper::Atom> atomvec;
  // Fill solute
  for (unsigned int i = 0; i < topo.num_solute_atoms(); i++) {
    clipper::Atom atm;
    assert(i < topo.xray_occupancies().size());
    atm.set_occupancy(topo.xray_occupancies()[i]);
    atm.set_coord_orth(clipper::Coord_orth(0.0, 0.0, 0.0));
    assert(i < topo.xray_b_factors().size());
    atm.set_u_iso(topo.xray_b_factors()[i] * 100.0 / sqpi2);
    DEBUG(1, "i: " << i << " size: " << topo.xray_elements().size());
    assert(i < topo.xray_elements().size());
    atm.set_element(topo.xray_elements()[i]);
    atomvec.push_back(atm);
  }
  // Fill solvent
  for (unsigned int i = 0; i < topo.num_solvent_atoms(); i++) {
    clipper::Atom atm;
    unsigned int index = i % topo.solvent(0).num_atoms();
    assert(index < topo.xray_solv_occupancies().size());
    atm.set_occupancy(topo.xray_solv_occupancies()[index]);
    atm.set_coord_orth(clipper::Coord_orth(0.0, 0.0, 0.0));
    assert(index < topo.xray_solv_b_factors().size());
    atm.set_u_iso(topo.xray_solv_b_factors()[index] * 100 / sqpi2);
    assert(index < topo.xray_solvelements().size());
    atm.set_element(topo.xray_solvelements()[index]);
    atomvec.push_back(atm);
  }
  atoms = clipper::Atom_list(atomvec);

  conf.special().xray_rest.resize(topo.xray_restraints().size());

  // Scale Fobs (needed for constant force-constant) -> scaled to sfscale
  const double sfscale = 100.0;
  double maxsf = 0.0;
  // Get max structure factor
  for (unsigned int i = 0; i < topo.xray_restraints().size(); i++) {
    if (maxsf < topo.xray_restraints()[i].sf)
      maxsf = topo.xray_restraints()[i].sf;
  }
  const double scalefactor = maxsf / sfscale;
  // scale
  for (unsigned int i = 0; i < topo.xray_restraints().size(); i++) {
    topo.xray_restraints()[i].sf /= scalefactor;
  }

  if (!quiet) {
    os.precision(2);
    os << "\nXRAYRESTINIT\n";
    os << "Restraint type              : ";
    switch (sim.param().xrayrest.xrayrest) {
      case simulation::xrayrest_off :
      {
        os << "No xray restraining\n";
        break;
      }
      case simulation::xrayrest_inst :
      {
        os << "Instantaneous xray restraining\n";
        break;
      }
      case simulation::xrayrest_avg :
      {
        os << "Time-averaged xray restraining\n";
        break;
      }
      case simulation::xrayrest_biq :
      {
        os << "Biquadratic time-averaged/instantaneous xray restraining\n";
        break;
      }
      case simulation::xrayrest_loel :
      {
        os << "Local-elevation xray restraining\n";
        break;
      }
      if (sim.param().xrayrest.readavg)
      os << "\treading xray averages from file\n";
    }
    os << "Restraint force-constant    : " << sim.param().xrayrest.force_constant << std::endl;
    os << "Spacegroup                  : " << sim.param().xrayrest.spacegroup << std::endl;
    os.precision(4);
    os << "Resolution                  : " << sim.param().xrayrest.resolution << std::endl;
    os << "Num experimental reflections: " << topo.xray_restraints().size() << std::endl;
    os << "Num expected reflections    : " << hkls.num_reflections() << std::endl;
    os << "Writeing electron density   : " << sim.param().xrayrest.writexmap << std::endl;
    os << "END\n\n";
  }
  // Check if too low resolution
  double expnorm = 0.0, calcnorm = 0.0, tempnorm = 0.0;
  for (unsigned int i = 0; i < topo.xray_restraints().size(); i++) {
    // calc max. experimental-index norm
    tempnorm = sqrt(double((topo.xray_restraints()[i].h * topo.xray_restraints()[i].h)+
            (topo.xray_restraints()[i].k * topo.xray_restraints()[i].k)+
            (topo.xray_restraints()[i].l * topo.xray_restraints()[i].l)));
    if (tempnorm > expnorm)
      expnorm = tempnorm;
  }
  for (int i = 0; i < hkls.num_reflections(); i++) {
    // calc max. calculation-index norm
    tempnorm = sqrt(double((hkls.hkl_of(i).h() * hkls.hkl_of(i).h())+
            (hkls.hkl_of(i).k() * hkls.hkl_of(i).k())+
            (hkls.hkl_of(i).l() * hkls.hkl_of(i).l())));
    if (tempnorm > calcnorm)
      calcnorm = tempnorm;
  }

  if (expnorm > calcnorm) {
    io::messages.add("Xray_restraint_interaction", "Too little reflections. Set higher resolution!", io::message::error);
    return 1;
  }
#endif
  return 0;
}


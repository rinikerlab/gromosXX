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

// special interactions
#include <interaction/interaction_types.h>

#include <interaction/special/xray_restraint_interaction.h>

#include <util/template_split.h>
#include <util/debug.h>
#include <vector>
#include <string>
#include <ios>
#include <time.h>

#ifdef OMP
#include <omp.h>
#endif

#undef MODULE
#undef SUBMODULE
#define MODULE interaction
#define SUBMODULE special
//#define HAVE_CLIPPER
interaction::Xray_Restraint_Interaction::Xray_Restraint_Interaction() : Interaction("XrayRestraint") {
}

interaction::Xray_Restraint_Interaction::~Xray_Restraint_Interaction() {
}

#ifdef HAVE_CLIPPER
/**
 * calculate the electron density from the model structure
 * @param[out] rho_calc the calculated electron density on a map
 * @param[in] atoms the list of the atoms
 */
void calculate_electron_density(clipper::Xmap<clipper::ftype32> & rho_calc,
        const clipper::Atom_list & atoms) {
  // this code is basically copied from the clipper library but parallelised
  // some hardcoded settings
  const double radius = 2.5;

  // zero the map
  rho_calc = 0.0;
  // create the range (size of atom)
  const clipper::Cell & cell = rho_calc.cell();
  const clipper::Grid_sampling & grid = rho_calc.grid_sampling();
  clipper::Grid_range gd(cell, grid, radius);

  const int atoms_size = atoms.size();

  // loop over atoms
#ifdef OMP
#pragma omp parallel for
#endif
  for (int i = 0; i < atoms_size; i++) {
    if (!atoms[i].is_null()) {
      clipper::AtomShapeFn sf(atoms[i].coord_orth(), atoms[i].element(),
              atoms[i].u_iso(), atoms[i].occupancy());
      // determine grad range of atom
      clipper::Coord_frac uvw = atoms[i].coord_orth().coord_frac(cell);
      clipper::Coord_grid g0 = uvw.coord_grid(grid) + gd.min();
      clipper::Coord_grid g1 = uvw.coord_grid(grid) + gd.max();

      // loop over atom's grid
      clipper::Xmap<clipper::ftype32>::Map_reference_coord i0, iu, iv, iw;
      i0 = clipper::Xmap<clipper::ftype32>::Map_reference_coord(rho_calc, g0);
      for (iu = i0; iu.coord().u() <= g1.u(); iu.next_u()) {
        for (iv = iu; iv.coord().v() <= g1.v(); iv.next_v()) {
          for (iw = iv; iw.coord().w() <= g1.w(); iw.next_w()) {
            // calculate the electron density and assign it to the gird point
            const double density = sf.rho(iw.coord_orth());
#ifdef OMP
#pragma omp critical
#endif
            rho_calc[iw] += density;
          }
        }
      } // loop over grid
    }
  } // loop over atoms
  // loop over the grid again and correct the multiplicity
  for (clipper::Xmap<clipper::ftype32>::Map_reference_index ix = rho_calc.first();
          !ix.last(); ix.next())
    rho_calc[ix] *= rho_calc.multiplicity(ix.coord());
}

/**
 * calculates the force from a reciprocal space difference map
 * @param[inout] D_k the reciprocal space difference map
 * @param[out] d_r the real space difference map
 * @param[in] atoms the list containing the atoms
 * @param[out] the force vector
 */
void calculate_force(clipper::FFTmap_p1 & D_k,
        clipper::Xmap<clipper::ftype32> & d_r,
        const clipper::Atom_list & atoms,
        math::VArray & force) {
  // these are just shortcuts to avoid many calls to the same functions
  const clipper::Spacegroup & spgr = d_r.spacegroup();

  // calculate the inverse symmetry operations of the spacegroup
  std::vector<clipper::Isymop> isymop;
  isymop.resize(spgr.num_symops());
  for(int j = 0; j < spgr.num_symops(); j++) {
    isymop[j] = clipper::Isymop(spgr.symop(j), d_r.grid_sampling());
  }
  const double volume = d_r.cell().volume();
  // convert from Angstrom to nm and add very annyoing scaling constants
  // to make the force volume AND resolution independent.
  const double scale = 10.0 / 2.0 * volume / (d_r.grid_sampling().size());
  // perform FFT of the difference map
  D_k.fft_h_to_x(scale);
  // loop over the (symmetry corrected map - even though this doesn't matter).
  for (clipper::Xmap<clipper::ftype32>::Map_reference_index ix = d_r.first(); !ix.last(); ix.next()) {
    // set initial data value
    const clipper::Coord_grid & coord = ix.coord();
    d_r[ix] = D_k.real_data(coord);
    // loop over symmetric copies of the grid point and add the data from these points
    for (int j = 1; j < spgr.num_symops(); j++) {
      d_r[ix] += D_k.real_data(coord.transform(isymop[j]).unit(D_k.grid_real()));
    }
    // correct for points mapped on themselves
    d_r[ix] /= d_r.multiplicity(coord);
  }

  // 3.5 is hardcoded atom radius for grid sampling.
  const double radius = 3.5;
  const clipper::Grid_sampling & grid = d_r.grid_sampling();
  // determine the range of the atomic electron density gradient on the gird
  clipper::Grid_range gd(d_r.cell(), grid, radius);

  // loop over the atoms - has to be int and not unsigned due to
  // stupid OpenMP rules
  const int atoms_size = atoms.size();
#ifdef OMP
#pragma omp parallel for
#endif
  for (int i = 0; i < atoms_size; i++) {
    if (!atoms[i].is_null()) {
      math::Vec gradient(0.0, 0.0, 0.0);
      clipper::AtomShapeFn sf(atoms[i].coord_orth(), atoms[i].element(),
              atoms[i].u_iso(), atoms[i].occupancy());

      // specify the derivatives we are interested in.
      sf.agarwal_params().resize(3);
      sf.agarwal_params()[0] = clipper::AtomShapeFn::X;
      sf.agarwal_params()[1] = clipper::AtomShapeFn::Y;
      sf.agarwal_params()[2] = clipper::AtomShapeFn::Z;
      // determine grid-ranges of this atom
      clipper::Coord_frac uvw = atoms[i].coord_orth().coord_frac(d_r.cell());
      clipper::Coord_grid g0 = uvw.coord_grid(grid) + gd.min();
      clipper::Coord_grid g1 = uvw.coord_grid(grid) + gd.max();
      clipper::Xmap<clipper::ftype64>::Map_reference_coord i0, iu, iv, iw;
      i0 = clipper::Xmap<clipper::ftype64>::Map_reference_coord(d_r, g0);
      std::vector<clipper::ftype> rho_grad(3, 0.0f);
      clipper::ftype temp_rho = 0.0f;
      // loop over grid and convolve with the atomic density gradient
      for (iu = i0; iu.coord().u() <= g1.u(); iu.next_u()) {
        for (iv = iu; iv.coord().v() <= g1.v(); iv.next_v()) {
          for (iw = iv; iw.coord().w() <= g1.w(); iw.next_w()) {
            // get gradient of the atomic electron density
            sf.rho_grad(iw.coord_orth(), temp_rho, rho_grad);

            // convolve it with difference map
            const double d_r_iw = d_r[iw];
            gradient(0) += d_r_iw * rho_grad[0];
            gradient(1) += d_r_iw * rho_grad[1];
            gradient(2) += d_r_iw * rho_grad[2];
          }
        }
      } // loop over map
      // add to force
      force(i) -= gradient;
      DEBUG(10, "grad(" << i << "): " << math::v2s(gradient));
    } // if atom not null
  } // for atoms
}


#endif
/**
 * calculate xray restraint interactions
 */
int interaction::Xray_Restraint_Interaction
::calculate_interactions(topology::Topology &topo,
        configuration::Configuration &conf,
        simulation::Simulation &sim) {
#ifdef HAVE_CLIPPER
  m_timer.start();
  // get number of atoms in simulation
  const int atoms_size = topo.num_atoms();
  // update clipper atomvec: convert the position to Angstrom
  for (int i = 0; i < atoms_size; i++) {
    atoms[i].set_coord_orth(clipper::Coord_orth(conf.current().pos(i)(0)*10.0,
            conf.current().pos(i)(1)*10.0,
            conf.current().pos(i)(2)*10.0));
  }
  // Calculate structure factors
  m_timer.start("structure factor");
  calculate_electron_density(rho_calc, atoms);
  // FFT the electron density to obtain the structure factors
  rho_calc.fft_to(fphi);

  m_timer.stop("structure factor");

  // sqr_calc:       sum of squared Fcalc
  // obs:            sum of Fobs
  // calc:           sum of Fcalc
  // obs_calc:       sum of Fobs*Fcalc
  // obs_calcavg:    sum of Fobs*Fcalc(averaged)
  // obs_k_calcavg:  sum of Fobs-k_avg*Fcalc(averaged)
  // obs_k_calc:     sum of Fobs-k*Fcalc
  // sqr_calcavg:    sum of squared time-averaged Fcalc
  // calcavg:        sum of time-averaged Fcalc
   m_timer.start("energy");
  // zero all the sums
  double sqr_calc = 0.0, obs = 0.0, calc = 0.0, obs_calc = 0.0, obs_k_calc = 0.0,
          sqr_calcavg = 0.0, calcavg = 0.0, obs_calcavg = 0.0, obs_k_calcavg = 0.0;
  // Number of reflections
  const unsigned int num_xray_rest = topo.xray_restraints().size();
  // e-term for time-average
  const double eterm = exp(-sim.time_step_size() / sim.param().xrayrest.tau);

  // loop over structure factors
  for (unsigned int i = 0; i < num_xray_rest; i++) {
    // filter calculated structure factors: save phases and amplitudes
    clipper::HKL hkl(topo.xray_restraints()[i].h, topo.xray_restraints()[i].k, topo.xray_restraints()[i].l);
    conf.special().xray_rest[i].sf_curr = fabs(fphi[hkl].f());
    conf.special().xray_rest[i].phase_curr = fphi[hkl].phi();
    DEBUG(15,"HKL:" << hkl.h() << "," << hkl.k() << "," << hkl.l()); 
    DEBUG(15,"\tSF: " << conf.special().xray_rest[i].sf_curr);

    // reset the averages at the beginning if requested
    if (!sim.param().xrayrest.readavg && sim.steps() == 0) {
      conf.special().xray_rest[i].sf_av = conf.special().xray_rest[i].sf_curr;
      conf.special().xray_rest[i].phase_av = conf.special().xray_rest[i].phase_curr;
    }

    // calculate averages
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
    return 1;
  }
#endif

  // calculate the scaling constants for inst and avg.
  double & k_inst = conf.special().xray.k_inst;
  k_inst = obs_calc / sqr_calc;
  double & k_avg = conf.special().xray.k_avg;
  k_avg = obs_calcavg / sqr_calcavg;
  DEBUG(10, "k_inst value: " << k_inst);
  DEBUG(10, "k_avg  value: " << k_avg);

  // calculate sums needed for R factors
  for (unsigned int i = 0; i < num_xray_rest; i++) {
    obs_k_calc += fabs(topo.xray_restraints()[i].sf - k_inst * conf.special().xray_rest[i].sf_curr);
    obs_k_calcavg += fabs(topo.xray_restraints()[i].sf - k_avg * conf.special().xray_rest[i].sf_av);
  }

  // calculate R factors: R_inst and R_avg
  double & R_inst = conf.special().xray.R_inst;
  R_inst = obs_k_calc / obs;
  double & R_avg = conf.special().xray.R_avg;
  R_avg = obs_k_calcavg / obs;
  DEBUG(10, "R_inst value: " << std::setw(15) << std::setprecision(8) << R_inst);
  DEBUG(10, "R_avg  value: " << std::setw(15) << std::setprecision(8) << R_avg);

  // calculate gradients
  // zero the reciprocal space difference map
  D_k.reset();

  double energy_sum = 0.0;
  // loop over retraints and calculate energy and difference map
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
        // Here, I tried to apply symmetry operations for non P1 spacegroups
        // but this had the effect the forces were not in agreement with
        // the finite difference result anymore. So we just safe the relection
        // given in the reflection list and not all symmetric copies. It's
        // up to the user to decide whether he should provide also the symmetric
        // copies for the refinement.
        D_k.set_hkl(hkl, clipper::data32::F_phi(sim.param().xrayrest.force_constant * dterm, conf.special().xray_rest[i].phase_curr));

        // save Fobs and PhiCalc for density maps. This will be corrected
        // for symmetry in the FFT step.
        fphi_obs.set_data(hkl, clipper::data32::F_phi(fobs/k_inst, conf.special().xray_rest[i].phase_curr));
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
        D_k.set_hkl(hkl, clipper::data32::F_phi(sim.param().xrayrest.force_constant * dterm, conf.special().xray_rest[i].phase_curr));

        fphi_obs.set_data(hkl, clipper::data32::F_phi(fobs/k_avg, conf.special().xray_rest[i].phase_av));
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
        D_k.set_hkl(hkl, clipper::data32::F_phi(sim.param().xrayrest.force_constant * dterm, conf.special().xray_rest[i].phase_curr));
        
        fphi_obs.set_data(hkl, clipper::data32::F_phi(xrs.sf/k_avg, conf.special().xray_rest[i].phase_av));
        break;
      }
      case simulation::xrayrest_loel :
      {
        break;
      }
    }
  }

  // finally calculate the energy
  conf.current().energies.xray_total = 0.5 * sim.param().xrayrest.force_constant * energy_sum;
  DEBUG(10, "energy: " << conf.current().energies.xray_total);
  m_timer.stop("energy");

  // start to calculate the forces
  m_timer.start("force");
  calculate_force(D_k, d_r, atoms, conf.current().force);
  m_timer.stop("force");
  m_timer.stop();

  // write xmap to external file
  if (sim.param().xrayrest.writexmap != 0 && sim.steps() % sim.param().xrayrest.writexmap == 0) {
    rho_calc.fft_from(fphi_obs);
    clipper::CCP4MAPfile mapfile;
    std::ostringstream file_name, asu_file_name;
    file_name << "density_frame_" << std::setw(int(log10(sim.param().step.number_of_steps)))
            << std::setfill('0') << sim.steps() << ".ccp4";
    if (sim.param().xrayrest.writedensity == 1 || sim.param().xrayrest.writedensity == 3) {
      mapfile.open_write(file_name.str());
      mapfile.export_xmap(rho_calc);
      mapfile.close_write();
    }
    // Non Cristallographic Map
    clipper::NXmap<double> asu(rho_calc.grid_asu(), rho_calc.operator_orth_grid());
    asu_file_name << "density_asu_frame_" << std::setw(int(log10(sim.param().step.number_of_steps)))
            << std::setfill('0') << sim.steps() << ".ccp4";
    if (sim.param().xrayrest.writedensity == 2 || sim.param().xrayrest.writedensity == 3) {
      mapfile.open_write(asu_file_name.str());
      mapfile.export_nxmap(asu);
      mapfile.close_write();
    }
  }
#endif

  return 0;
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

    // create a grid and a crystalographic map
  const double shannon_rate = 1.5;
  const clipper::Grid_sampling grid_rho_calc(spacegr, cell, reso, shannon_rate);
  rho_calc.init(spacegr, cell, grid_rho_calc);
  const clipper::Grid_sampling grid_rho_obs(spacegr, cell, reso, shannon_rate);
  rho_obs.init(spacegr, cell, grid_rho_obs);

  hkls.init(spacegr, cell, reso, true);
  fphi.init(hkls, hkls.cell());
  fphi_obs.init(hkls, hkls.cell());

  // The difference map has to be a P 1 map in order to get agreement with
  // the finite difference results. However, the reasons for this are
  // not 100% clear. In principle (from theory) a spacegroup depdendent
  // Xmap should do the job.
  clipper::Spgr_descr spgrinit(clipper::String("P 1"), clipper::Spgr_descr::HM);
  clipper::Spacegroup p1_spacegr;
  p1_spacegr.init(spgrinit);
  // create a grid and a P1 FFT map. Here we can use the FFTmap_p1 which was
  // designed for fast P1.
  // 1.5 is shannon-rate for oversampled FFT
  const clipper::Grid_sampling fftgrid(p1_spacegr, cell, reso, 1.5);
  D_k.init(fftgrid);
  // we still need an Xmap for convenient looping over the data in order
  // to do the convolution.
  const clipper::Grid_sampling grid_d_r(spacegr, cell, reso, 1.5);
  d_r.init(spacegr, cell, grid_d_r);

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


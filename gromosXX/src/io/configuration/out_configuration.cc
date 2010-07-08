/**
 * @file out_configuration.cc
 * definition of the Out_Configuration methods.
 */
#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <simulation/parameter.h>
#include <configuration/configuration.h>
#include <configuration/energy.h>

#include <math/periodicity.h>
#include <math/volume.h>
#include <math/transformation.h>

#include <io/print_block.h>
#include <io/argument.h>

#include "out_configuration.h"

#include <util/replica_data.h>
#include <util/template_split.h>
#include <util/debug.h>
#include <limits>
#include <util/umbrella_weight.h>

#undef MODULE
#undef SUBMODULE

#define MODULE io
#define SUBMODULE configuration

// declarations
static void _print_energyred_helper(std::ostream & os, configuration::Energy const &e);

static void _print_volumepressurered_helper(std::ostream &os,
        double mass,
        double const & phi, double const & theta, double const &psi,
        simulation::Multibath const & m,
        std::vector<double> const & s,
        configuration::Energy const & e,
        math::Box const & b,
        math::boundary_enum t,
        math::Matrix const & p,
        math::Matrix const & v,
        math::Matrix const & k);

io::Out_Configuration::Out_Configuration(std::string title,
        std::ostream & os)
: m_has_replica_traj(false),
m_output(os),
m_final(false),
m_replica(false),
m_every_pos(0),
m_every_vel(0),
m_every_force(0),
m_every_energy(0),
m_every_free_energy(0),
m_every_blockaverage(0),
m_every_ramd(0),
m_every_cos_pos(0),
m_every_jvalue(0),
m_every_xray(0),
m_every_disres(0),
m_every_dat(0),
m_every_leus(0),
m_write_blockaverage_energy(false),
m_write_blockaverage_free_energy(false),
m_precision(9),
m_force_precision(9),
m_distance_restraint_precision(7),
m_width(15),
m_force_width(18),
m_title(title),
minimum_energy(std::numeric_limits<double>::max()) {
  _print_title(m_title, "output file", os);
}

io::Out_Configuration::~Out_Configuration() 
{
  // std::cout << "out_configuration destructor" << std::endl;

  if (m_every_pos) {
    m_pos_traj.flush();
    m_pos_traj.close();
  }

  if (m_final) {
    m_final_conf.flush();
    m_final_conf.close();
  }

  if (m_every_vel) {
    m_vel_traj.flush();
    m_vel_traj.close();
  }

  if (m_every_force) {
    m_force_traj.flush();
    m_force_traj.close();
  }

  if (m_every_energy) {
    m_energy_traj.flush();
    m_energy_traj.close();
  }

  if (m_has_replica_traj) {
    m_replica_traj.flush();
    m_replica_traj.close();
  }

  if (m_every_free_energy) {
    m_free_energy_traj.flush();
    m_free_energy_traj.close();
  }

  if (m_write_blockaverage_energy) {
    m_blockaveraged_energy.flush();
    m_blockaveraged_energy.close();
  }

  if (m_write_blockaverage_free_energy) {
    m_blockaveraged_free_energy.flush();
    m_blockaveraged_free_energy.close();
  }

  if (m_every_ramd) {
    m_ramd_traj.flush();
    m_ramd_traj.close();
  }

  if (m_every_cos_pos || m_every_jvalue || m_every_xray || m_every_disres || m_every_dat || m_every_leus) { // add others if there are any
    m_special_traj.flush();
    m_special_traj.close();
  }
}

void io::Out_Configuration::_print_title(std::string title,
        std::string name,
        std::ostream &os) {
  os << "TITLE\n\t"
          << title << "\n"
          << "\t" << name
          << "\nEND\n";
}

void io::Out_Configuration::init(io::Argument & args,
        simulation::Parameter const & param) {
  if (args.count(argname_fin) > 0)
    final_configuration(args[argname_fin]);
  else io::messages.add("argument " + argname_fin + " for final configuration required!",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_trj) > 0)
    trajectory(args[argname_trj], param.write.position);
  else if (param.write.position)
    io::messages.add("write trajectory but no " + argname_trj + " argument",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_trv) > 0)
    velocity_trajectory(args[argname_trv], param.write.velocity);
  else if (param.write.velocity)
    io::messages.add("write velocity trajectory but no trv argument",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_trf) > 0)
    force_trajectory(args[argname_trf], param.write.force);
  else if (param.write.force)
    io::messages.add("write force trajectory but no trf argument",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_trs) > 0)
    special_trajectory(args[argname_trs], param.polarise.write, param.jvalue.write,
            param.xrayrest.write, param.distanceres.write, param.print.monitor_dihedrals,
            param.localelev.write);
  else if (param.polarise.write || param.jvalue.write || param.xrayrest.write 
        || param.distanceres.write || param.print.monitor_dihedrals || param.localelev.write)
    io::messages.add("write special trajectory but no trs argument",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_re) > 0)
    replica_trajectory(args[argname_re]);

  if (args.count(argname_tre) > 0)
    energy_trajectory(args[argname_tre], param.write.energy);
  else if (param.write.energy)
    io::messages.add("write energy trajectory but no " + argname_tre + " argument",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_trg) > 0)
    free_energy_trajectory(args[argname_trg], param.write.free_energy);
  else if (param.write.free_energy)
    io::messages.add("write free energy trajectory but no trg argument",
          "Out_Configuration",
          io::message::error);

  if (args.count(argname_bae) > 0)
    block_averaged_energy(args[argname_bae], param.write.block_average);
  else if (param.write.block_average && param.write.energy)
    io::messages.add("write block averaged energy but no bae argument",
          "Out_Configuration",
          io::message::error);

  if (param.perturbation.perturbation) {
    if (args.count(argname_bag) > 0)
      block_averaged_free_energy(args[argname_bag],
            param.write.block_average);
    else if (param.write.block_average && param.write.free_energy)
      io::messages.add("write block averaged free energy "
            "but no bag argument",
            "Out_Configuration",
            io::message::error);
  }
  if (args.count(argname_tramd) > 0)
    ramd_trajectory(args[argname_tramd], param.ramd.every);
  else if (param.ramd.fc != 0.0 && param.ramd.every)
    io::messages.add("write RAMD trajectory but no tramd argument",
          "Out_Configuration",
          io::message::error);

  if (param.replica.num_T * param.replica.num_l) {
    m_replica = true;
  }

}

void io::Out_Configuration::write(configuration::Configuration &conf,
        topology::Topology const &topo,
        simulation::Simulation const &sim,
        output_format const form) {
  // standard trajectories

  bool constraint_force = sim.param().constraint.solute.algorithm == simulation::constr_shake ||
          sim.param().constraint.solvent.algorithm == simulation::constr_shake;

  // check whether a new energy minimum was found
  bool minimum_found = false;
  if (sim.param().write.energy_index > 0) {
    double current_energy = conf.old().energies.get_energy_by_index(sim.param().write.energy_index);

    // found a new minimum?
    if (current_energy < minimum_energy) {
      minimum_found = true;
      minimum_energy = current_energy;
    }
  }

  if (form == reduced) {
    /**
     * set this to true when you print the timestep to the special traj.
     * make sure you don't print it twice. 
     */
    bool special_timestep_printed = false;

    if (m_every_pos && ((sim.steps() % m_every_pos) == 0 || minimum_found)) {
      // don't write starting configuration if analyzing a trajectory
      if (sim.steps() || !sim.param().analyze.analyze) {
        _print_timestep(sim, m_pos_traj);

        if (sim.param().write.position_solute_only)
          _print_positionred(conf, topo, topo.num_solute_atoms(), m_pos_traj);
        else
          _print_positionred(conf, topo, topo.num_atoms(), m_pos_traj);

        if (conf.boundary_type != math::vacuum)
          _print_box(conf, m_pos_traj);

        m_pos_traj.flush();
      }
      // a new block begins. let's reset the minimum
      minimum_energy = conf.old().energies.get_energy_by_index(sim.param().write.energy_index);
    }

    if (m_every_vel && (sim.steps() % m_every_vel) == 0) {
      _print_timestep(sim, m_vel_traj);
      if (sim.param().write.velocity_solute_only)
        _print_velocityred(conf, topo.num_solute_atoms(), m_vel_traj);
      else
        _print_velocityred(conf, topo.num_atoms(), m_vel_traj);
      m_vel_traj.flush();
    }

    if (m_every_force && ((sim.steps() + 1) % m_every_force) == 0) {
      if (sim.steps()) {
        _print_old_timestep(sim, m_force_traj);
        if (sim.param().write.force_solute_only)
          _print_forcered(conf, topo.num_solute_atoms(), m_force_traj, constraint_force);
        else
          _print_forcered(conf, topo.num_atoms(), m_force_traj, constraint_force);
        m_force_traj.flush();
      }
    }

    if (m_every_cos_pos && (sim.steps() % m_every_cos_pos) == 0) {
      if (!special_timestep_printed) {
        _print_timestep(sim, m_special_traj);
        special_timestep_printed = true;
      }
      _print_cos_position(conf, topo, m_special_traj);
      m_special_traj.flush();
    }

    if (m_every_jvalue && sim.steps() && (sim.steps() % m_every_jvalue) == 0) {
      if (!special_timestep_printed) {
        _print_timestep(sim, m_special_traj);
        special_timestep_printed = true;
      }
      _print_jvalue(sim.param(), conf, topo, m_special_traj, true);
      m_special_traj.flush();
    }

    if (m_every_xray && sim.steps() && (sim.steps() % m_every_xray) == 0) {
      if (!special_timestep_printed) {
        _print_timestep(sim, m_special_traj);
        special_timestep_printed = true;
      }
      _print_xray_rvalue(sim.param(), conf, m_special_traj);
      _print_xray_umbrellaweightthresholds(sim.param(), topo, m_special_traj);
      //_print_xray(sim.param(), conf, topo, m_special_traj);
      m_special_traj.flush();
    }

    if (m_every_disres && sim.steps() && (sim.steps() % m_every_disres) == 0) {
      if (!special_timestep_printed) {
        _print_timestep(sim, m_special_traj);
        special_timestep_printed = true;
      }
      _print_distance_restraints(conf, topo, m_special_traj);
      m_special_traj.flush();
    }

    if (m_every_dat) {
      if (!special_timestep_printed) {
        _print_timestep(sim, m_special_traj);
        special_timestep_printed = true;
      }
      _print_dihangle_trans(conf, topo, m_special_traj);
      m_special_traj.flush();
    }
    if (m_every_leus && sim.steps() && (sim.steps() % m_every_leus) == 0) {
      if (!special_timestep_printed) {
        _print_timestep(sim, m_special_traj);
        special_timestep_printed = true;
      }
      _print_umbrellas(conf, m_special_traj);
      m_special_traj.flush();
    }


    if (m_every_energy && (((sim.steps() - 1) % m_every_energy) == 0 || minimum_found)) {
      if (sim.steps()) {
        _print_old_timestep(sim, m_energy_traj);
        _print_energyred(conf, m_energy_traj);
        _print_volumepressurered(topo, conf, sim, m_energy_traj);
        m_energy_traj.flush();
      }
    }

    if (m_every_free_energy && ((sim.steps() - 1) % m_every_free_energy) == 0) {
      if (sim.steps()) {
        _print_old_timestep(sim, m_free_energy_traj);
        _print_free_energyred(conf, topo, m_free_energy_traj);
        m_free_energy_traj.flush();
      }
    }

    if (m_every_blockaverage && ((sim.steps() - 1) % m_every_blockaverage) == 0) {

      if (m_write_blockaverage_energy) {
        if (sim.steps()) {
          _print_old_timestep(sim, m_blockaveraged_energy);
          _print_blockaveraged_energyred(conf, m_blockaveraged_energy);
          _print_blockaveraged_volumepressurered(conf, sim, m_blockaveraged_energy);
          m_blockaveraged_energy.flush();
        }
      }

      if (m_write_blockaverage_free_energy) {
        if (sim.steps()) {
          _print_old_timestep(sim, m_blockaveraged_free_energy);
          _print_blockaveraged_free_energyred(conf, sim.param().perturbation.dlamt,
                  m_blockaveraged_free_energy);
          m_blockaveraged_free_energy.flush();
        }
      }
      conf.current().averages.block().zero();
    }

    if (m_every_ramd && (sim.steps() % m_every_ramd) == 0) {
      _print_timestep(sim, m_ramd_traj);
      _print_ramd(topo, conf, sim, m_ramd_traj);
      m_ramd_traj.flush();
    }

  } else if (form == final && m_final) {
    _print_timestep(sim, m_final_conf);
    _print_position(conf, topo, m_final_conf);
    _print_lattice_shifts(conf, topo, m_final_conf);

    if (sim.param().polarise.cos)
      _print_cos_position(conf, topo, m_final_conf);

    if (sim.param().minimise.ntem == 0)
      _print_velocity(conf, topo, m_final_conf);

    _print_box(conf, m_final_conf);

    if (sim.param().constraint.solute.algorithm
            == simulation::constr_flexshake) {
      _print_flexv(conf, topo, m_final_conf);
    }

    if (sim.param().stochastic.sd) {
      _print_stochastic_integral(conf, topo, m_final_conf);
    }

    if (sim.param().perturbation.perturbation) {
      _print_pertdata(topo, m_final_conf);
    }

    if (sim.param().distanceres.distanceres < 0) {
      _print_distance_restraint_averages(conf, topo, m_final_conf);
    }

    if (sim.param().posrest.posrest != simulation::posrest_off) {
      _print_position_restraints(sim, topo, conf, m_final_conf);
    }

    if (sim.param().jvalue.mode != simulation::jvalue_restr_off) {
      _print_jvalue(sim.param(), conf, topo, m_final_conf, false);
    }

    if (sim.param().xrayrest.xrayrest != simulation::xrayrest_off) {
      _print_xray(sim.param(), conf, topo, m_final_conf, /*final=*/ true);
      _print_xray_umbrellaweightthresholds(sim.param(), topo, m_final_conf);
    }

    if (sim.param().localelev.localelev != simulation::localelev_off) {
      _print_umbrellas(conf, m_final_conf);
    }

    if (sim.param().rottrans.rottrans) {
      _print_rottrans(conf, sim, m_final_conf);
    }

    if (sim.param().pscale.jrest) {
      _print_pscale_jrest(conf, topo, m_final_conf);
    }

    if (sim.param().multibath.nosehoover > 1) {
      _print_nose_hoover_chain_variables(sim.multibath(), m_final_conf);
    }

    // forces and energies still go to their trajectories
    if (m_every_force && ((sim.steps() - 1) % m_every_force) == 0) {
      _print_old_timestep(sim, m_force_traj);
      if (sim.param().write.force_solute_only)
        _print_forcered(conf, topo.num_solute_atoms(), m_force_traj, constraint_force);
      else
        _print_forcered(conf, topo.num_atoms(), m_force_traj, constraint_force);
    }

    if (m_every_energy && ((sim.steps() - 1) % m_every_energy) == 0) {
      _print_old_timestep(sim, m_energy_traj);
      _print_energyred(conf, m_energy_traj);
      _print_volumepressurered(topo, conf, sim, m_energy_traj);
    }

    if (m_every_free_energy && ((sim.steps() - 1) % m_every_free_energy) == 0) {
      _print_old_timestep(sim, m_free_energy_traj);
      _print_free_energyred(conf, topo, m_free_energy_traj);
    }

    if (m_every_blockaverage && ((sim.steps() - 1) % m_every_blockaverage) == 0) {

      if (m_write_blockaverage_energy) {
        if (sim.steps()) {
          _print_old_timestep(sim, m_blockaveraged_energy);
          _print_blockaveraged_energyred(conf, m_blockaveraged_energy);
          _print_blockaveraged_volumepressurered(conf, sim, m_blockaveraged_energy);
        }
      }

      if (m_write_blockaverage_free_energy) {
        if (sim.steps()) {
          _print_old_timestep(sim, m_blockaveraged_free_energy);
          _print_blockaveraged_free_energyred(conf, sim.param().perturbation.dlamt,
                  m_blockaveraged_free_energy);
        }
      }
      conf.current().averages.block().zero();
    }

    if (conf.special().shake_failure_occurred) {
      _print_shake_failure(conf, topo, m_final_conf);
    }

  } else {

    // not reduced or final (so: decorated)

    if (m_every_pos && (sim.steps() % m_every_pos) == 0) {
      _print_timestep(sim, m_pos_traj);
      _print_position(conf, topo, m_pos_traj);
      if (conf.boundary_type != math::vacuum)
        _print_box(conf, m_pos_traj);
    }

    if (m_every_vel && (sim.steps() % m_every_vel) == 0) {
      _print_timestep(sim, m_vel_traj);
      _print_velocity(conf, topo, m_vel_traj);
    }

    if (m_every_force && (sim.steps() % m_every_force) == 0) {
      if (sim.steps()) {
        _print_timestep(sim, m_force_traj);
        _print_force(conf, topo, m_force_traj, constraint_force);
      }
    }
  }

  // done writing!

}

/**
 * write out replicas
 */
void io::Out_Configuration::write_replica
(
        std::vector<util::Replica_Data> & replica_data,
        std::vector<configuration::Configuration> & conf,
        topology::Topology const & topo,
        simulation::Simulation const &sim,
        output_format const form) {
  // standard trajectories
  if (form == reduced) {

    if (m_every_pos && (sim.steps() % m_every_pos) == 0) {
      _print_timestep(sim, m_pos_traj);
      _print_replica_information(replica_data, m_pos_traj);

      for (unsigned int i = 0; i < conf.size(); ++i) {
        _print_positionred(conf[i], topo, topo.num_atoms(), m_pos_traj);
        _print_velocityred(conf[0], topo.num_atoms(), m_vel_traj);

        if (conf[i].boundary_type != math::vacuum)
          _print_box(conf[i], m_pos_traj);
      }
    }
  } else if (form == final && m_final) {
    for (unsigned int i = 0; i < conf.size(); ++i) {

      m_final_conf << "REPLICAFRAME\n"
              << std::setw(12) << i + 1
              << "\nEND\n";

      if (i == 0) {
        _print_timestep(sim, m_final_conf);
        _print_replica_information(replica_data, m_final_conf);
      }

      _print_position(conf[i], topo, m_final_conf);
      if (sim.param().polarise.cos)
        _print_cos_position(conf[i], topo, m_final_conf);
      _print_velocity(conf[i], topo, m_final_conf);
      _print_lattice_shifts(conf[i], topo, m_final_conf);
      _print_box(conf[i], m_final_conf);
      if (conf[i].special().shake_failure_occurred) {
        _print_shake_failure(conf[i], topo, m_final_conf);
      }
    }

    /*
    if(sim.param().jvalue.mode != simulation::restr_off){
      _print_jvalue(sim.param(), conf[0], topo, m_final_conf);
    }
    if(sim.param().pscale.jrest){
      _print_pscale_jrest(conf[0], topo, m_final_conf);
    }
     */

    if (sim.param().multibath.nosehoover > 1) {
      _print_nose_hoover_chain_variables(sim.multibath(), m_final_conf);
    }
  }
  // done writing replicas!
}

void io::Out_Configuration
::final_configuration(std::string name) {
  m_final_conf.open(name.c_str());

  _print_title(m_title, "final configuration", m_final_conf);
  m_final = true;
}

void io::Out_Configuration
::trajectory(std::string name, int every) {
  m_pos_traj.open(name.c_str());

  m_every_pos = every;
  _print_title(m_title, "position trajectory", m_pos_traj);
}

void io::Out_Configuration
::velocity_trajectory(std::string name, int every) {
  m_vel_traj.open(name.c_str());

  m_every_vel = every;
  _print_title(m_title, "velocity trajectory", m_vel_traj);
}

void io::Out_Configuration
::force_trajectory(std::string name, int every) {
  m_force_traj.open(name.c_str());

  m_every_force = every;
  _print_title(m_title, "force trajectory", m_force_traj);
}

void io::Out_Configuration
::special_trajectory(std::string name, int every_cos, int every_jvalue, 
                     int every_xray, int every_disres, int every_dat, int every_leus) {

  m_special_traj.open(name.c_str());

  m_every_cos_pos = every_cos;
  m_every_jvalue = every_jvalue;
  m_every_xray = every_xray;
  m_every_disres = every_disres;
  m_every_dat = every_dat;
  m_every_leus = every_leus;
  _print_title(m_title, "special trajectory", m_special_traj);
}

void io::Out_Configuration
::energy_trajectory(std::string name, int every) {
  m_energy_traj.open(name.c_str());

  m_every_energy = every;
  _print_title(m_title, "energy trajectory", m_energy_traj);
}

void io::Out_Configuration
::replica_trajectory(std::string name) {
  m_replica_traj.open(name.c_str());

  m_has_replica_traj = true;
  _print_title(m_title, "replica trajectory", m_replica_traj);
}

void io::Out_Configuration
::free_energy_trajectory(std::string name, int every) {
  m_free_energy_traj.open(name.c_str());

  m_every_free_energy = every;
  _print_title(m_title, "free energy trajectory", m_free_energy_traj);
}

void io::Out_Configuration
::block_averaged_energy(std::string name, int every) {
  m_blockaveraged_energy.open(name.c_str());

  if (m_every_blockaverage && m_every_blockaverage != every) {
    io::messages.add("overwriting how often block averages are written out illegal",
            "Out_Configuration",
            io::message::error);
  }
  m_every_blockaverage = every;
  m_write_blockaverage_energy = true;
  _print_title(m_title, "block averaged energies", m_blockaveraged_energy);
}

void io::Out_Configuration
::block_averaged_free_energy(std::string name, int every) {
  m_blockaveraged_free_energy.open(name.c_str());

  if (m_every_blockaverage && m_every_blockaverage != every) {
    io::messages.add("overwriting how often block averages are written out illegal",
            "Out_Configuration",
            io::message::error);
  }
  m_every_blockaverage = every;
  m_write_blockaverage_free_energy = true;
  _print_title(m_title, "block averaged free energies", m_blockaveraged_free_energy);
}

void io::Out_Configuration
::ramd_trajectory(std::string name, int every) {
  m_ramd_traj.open(name.c_str());

  m_every_ramd = every;
  _print_title(m_title, "RAMD trajectory", m_ramd_traj);
}

void io::Out_Configuration
::_print_timestep(simulation::Simulation const &sim,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "TIMESTEP\n"
          << std::setw(m_width) << sim.steps()
          << " "
          << std::setw(m_width - 1) << sim.time()
          << "\nEND\n";
}

void io::Out_Configuration
::_print_old_timestep(simulation::Simulation const &sim,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "TIMESTEP\n"
          << std::setw(m_width) << sim.steps() - 1
          << " "
          << std::setw(m_width - 1) << sim.time() - sim.time_step_size()
          << "\nEND\n";

}

template<math::boundary_enum b>
void _print_g96_position_bound(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os, int width,
        bool old_conf = false) {
  const configuration::Configuration::state_struct * state;
  if (old_conf)
    state = & conf.old();
  else
    state = & conf.current();

  math::Periodicity<b> periodicity(state->box);
  math::VArray const &pos = state->pos;
  topology::Solute const &solute = topo.solute();
  std::vector<std::string> const &residue_name = topo.residue_names();


  math::Vec v, v_box, trans, r;
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct) {
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  }

  os << "# first 24 chars ignored\n";

  // put chargegroups into the box (on the fly)
  topology::Chargegroup_Iterator cg_it = topo.chargegroup_begin(),
          cg_to = topo.chargegroup_end();

  // solute chargegroups...
  unsigned int i = 0;
  for (; i < topo.num_solute_chargegroups(); ++cg_it, ++i) {
    // gather on first atom...
    v = pos(*cg_it.begin());
    v_box = v;
    periodicity.put_into_positive_box(v_box);
    trans = v_box - v;

    // atoms in a chargegroup
    topology::Atom_Iterator at_it = cg_it.begin(),
            at_to = cg_it.end();

    for (; at_it != at_to; ++at_it) {
      r = pos(*at_it) + trans;
      //rotate to original Cartesian coordinates
      r = math::Vec(math::product(Rmat, r));
      os << std::setw(5) << solute.atom(*at_it).residue_nr + 1 << " "
              << std::setw(5) << std::left
              << residue_name[solute.atom(*at_it).residue_nr] << " "
              << std::setw(6) << std::left << solute.atom(*at_it).name
              << std::right
              << std::setw(6) << *at_it + 1
              << std::setw(width) << r(0)
              << std::setw(width) << r(1)
              << std::setw(width) << r(2)
              << "\n";
    }
  }

  // solvent chargegroups
  unsigned int s = 0;
  unsigned int mol = 0;

  for (; cg_it != cg_to; ++cg_it, ++mol) {
    v = pos(**cg_it);
    v_box = v;
    periodicity.put_into_positive_box(v_box);
    trans = v_box - v;

    if (mol >= topo.num_solvent_molecules(s))++s;

    // loop over the atoms
    topology::Atom_Iterator at_it = cg_it.begin(),
            at_to = cg_it.end();
    // one chargegroup per solvent
    for (unsigned int atom = 0; at_it != at_to; ++at_it, ++atom) {
      r = pos(*at_it) + trans;
      //rotate to original Cartesian coordinates
      r = math::Vec(math::product(Rmat, r));
      os << std::setw(5) << mol + 1
              << ' ' << std::setw(5) << std::left
              << residue_name[topo.solvent(s).atom(atom).residue_nr] << " "
              << std::setw(6) << std::left << topo.solvent(s).atom(atom).name
              << std::right
              << std::setw(6) << *at_it + 1
              << std::setw(width) << r(0)
              << std::setw(width) << r(1)
              << std::setw(width) << r(2)
              << "\n";
    }
  }

}

/**
 * i need a specialized function to put the particles into the box.
 */
template<math::boundary_enum b>
void _print_position_bound(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os, int width) {
  math::Periodicity<b> periodicity(conf.current().box);
  math::VArray const &pos = conf.current().pos;
  topology::Solute const &solute = topo.solute();
  std::vector<std::string> const &residue_name = topo.residue_names();

  math::Vec v;
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct) {
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  }

  os << "# first 24 chars ignored\n";

  for (int i = 0, to = topo.num_solute_atoms(); i < to; ++i) {

    v = pos(i);
    periodicity.put_into_box(v);
    //rotate to original Cartesian coordinates
    v = math::Vec(math::product(Rmat, v));
    os << std::setw(5) << solute.atom(i).residue_nr + 1 << " "
            << std::setw(5) << std::left
            << residue_name[solute.atom(i).residue_nr] << " "
            << std::setw(6) << std::left << solute.atom(i).name << std::right
            << std::setw(6) << i + 1
            << std::setw(width) << v(0)
            << std::setw(width) << v(1)
            << std::setw(width) << v(2)
            << "\n";
  }

  int index = topo.num_solute_atoms();
  int res_nr = 1;

  for (unsigned int s = 0; s < topo.num_solvents(); ++s) {

    for (unsigned int m = 0; m < topo.num_solvent_molecules(s); ++m, ++res_nr) {

      for (unsigned int a = 0; a < topo.solvent(s).num_atoms(); ++a, ++index) {

        v = pos(index);
        periodicity.put_into_positive_box(v);
        //rotate to original Cartesian coordinates
        v = math::Vec(math::product(Rmat, v));
        os << std::setw(5) << res_nr
                << ' ' << std::setw(5) << std::left
                << residue_name[topo.solvent(s).atom(a).residue_nr] << " "
                << std::setw(6) << std::left
                << topo.solvent(s).atom(a).name << std::right
                << std::setw(6) << index + 1
                << std::setw(width) << v(0)
                << std::setw(width) << v(1)
                << std::setw(width) << v(2)
                << "\n";
      }
    }
  }
}

void io::Out_Configuration
::_print_position(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "POSITION\n";

  SPLIT_BOUNDARY(_print_g96_position_bound,
          conf, topo, os, m_width);

  os << "END\n";

}

void io::Out_Configuration
::_print_shake_failure(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "SHAKEFAILPOSITION\n";
  SPLIT_BOUNDARY(_print_g96_position_bound,
          conf, topo, os, m_width);
  os << "END\n";
  os << "SHAKEFAILPREVPOSITION\n";
  SPLIT_BOUNDARY(_print_g96_position_bound,
          conf, topo, os, m_width, true /* old coords */);
  os << "END\n";

}

template<math::boundary_enum b>
void _print_g96_positionred_bound(configuration::Configuration const &conf,
        topology::Topology const &topo,
        int num,
        std::ostream &os, int width) {
  DEBUG(10, "g96 positionred");

  math::Periodicity<b> periodicity(conf.current().box);
  math::VArray pos = conf.current().pos;

  math::Vec v, v_box, trans, r;
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));

  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));

  assert(num >= 0);

  // put chargegroups into the box (on the fly)
  topology::Chargegroup_Iterator cg_it = topo.chargegroup_begin(),
          cg_to = topo.chargegroup_end();
  DEBUG(10, "cg to : " << **cg_to << std::endl);

  // solute chargegroups...
  unsigned int i = 0, count = 0;
  for (; i < topo.num_solute_chargegroups(); ++cg_it, ++i) {
    DEBUG(10, "solute cg: " << i);
    // gather on first atom...
    v = pos(*cg_it.begin());
    v_box = v;
    periodicity.put_into_positive_box(v_box);
    trans = v_box - v;

    // atoms in a chargegroup
    topology::Atom_Iterator at_it = cg_it.begin(),
            at_to = cg_it.end();

    for (; at_it != at_to; ++at_it, ++count) {

      if (*at_it >= unsigned(num)) return;

      DEBUG(10, "atom: " << count);
      r = pos(*at_it) + trans;
      //rotate back to original Carthesian Coord
      r = math::Vec(math::product(Rmat, r));
      os << std::setw(width) << r(0)
              << std::setw(width) << r(1)
              << std::setw(width) << r(2)
              << "\n";

      if ((count + 1) % 10 == 0) os << '#' << std::setw(10) << count + 1 << "\n";

    }
  }

  DEBUG(10, "solvent");

  // solvent chargegroups
  unsigned int mol = 0;

  for (; cg_it != cg_to; ++cg_it, ++mol) {
    DEBUG(10, "solvent " << mol);

    v = pos(**cg_it);
    v_box = v;
    periodicity.put_into_positive_box(v_box);
    trans = v_box - v;

    // loop over the atoms
    topology::Atom_Iterator at_it = cg_it.begin(),
            at_to = cg_it.end();
    // one chargegroup per solvent
    for (; at_it != at_to; ++at_it, ++count) {
      DEBUG(10, "\tatom " << count);

      if (*at_it >= unsigned(num)) return;

      r = pos(*at_it) + trans;
      //rotate back to original Carthesian Coord
      r = math::Vec(math::product(Rmat, r));
      os << std::setw(width) << r(0)
              << std::setw(width) << r(1)
              << std::setw(width) << r(2)
              << "\n";

      if ((count + 1) % 10 == 0) os << '#' << std::setw(10) << count + 1 << "\n";

    }
  }
}

template<math::boundary_enum b>
void
_print_positionred_bound(configuration::Configuration const &conf,
        int num,
        std::ostream &os, int width) {
  math::Periodicity<b> periodicity(conf.current().box);

  math::VArray const &pos = conf.current().pos;
  math::Vec v;
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));

  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));

  DEBUG(10, "writing POSITIONRED " << pos.size());

  for (int i = 0; i < num; ++i) {

    v = pos(i);
    periodicity.put_into_box(v);
    //rotate back into original Cartesian coordinates
    v = math::Vec(math::product(Rmat, v));
    os << std::setw(width) << v(0)
            << std::setw(width) << v(1)
            << std::setw(width) << v(2)
            << "\n";

    if ((i + 1) % 10 == 0) os << '#' << std::setw(10) << i + 1 << "\n";
  }

}

inline void io::Out_Configuration
::_print_positionred(configuration::Configuration const &conf,
        topology::Topology const &topo,
        int num,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "POSITIONRED\n";
  DEBUG(7, "configuration boundary type :" << conf.boundary_type);

  SPLIT_BOUNDARY(_print_g96_positionred_bound, conf, topo, num, os, m_width);

  os << "END\n";

}

inline void io::Out_Configuration
::_print_cos_position(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "COSDISPLACEMENTS\n";

  math::VArray const &posV = conf.current().posV;
  //rotate back to original carthesian coordinates
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct) {
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  }
  math::Vec posV_rot;
  for (unsigned int i = 0; i < posV.size(); ++i) {
    posV_rot = math::Vec(math::product(Rmat, posV(i)));
    os << std::setw(m_width) << posV_rot(0)
            << std::setw(m_width) << posV_rot(1)
            << std::setw(m_width) << posV_rot(2)
            << "\n";

    if ((i + 1) % 10 == 0) os << '#' << std::setw(10) << i + 1 << "\n";
  }

  os << "END\n";
}

void io::Out_Configuration
::_print_velocity(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "VELOCITY\n";

  math::VArray const &vel = conf.current().vel;
  topology::Solute const &solute = topo.solute();
  std::vector<std::string> const &residue_name = topo.residue_names();

  os << "# first 24 chars ignored\n";
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));

  math::Vec vel_rot;
  for (int i = 0, to = topo.num_solute_atoms(); i < to; ++i) {
    //rotate back to original Carthesian coordinates
    vel_rot = math::Vec(math::product(Rmat, vel(i)));
    os << std::setw(5) << solute.atom(i).residue_nr + 1 << " "
            << std::setw(5) << std::left << residue_name[solute.atom(i).residue_nr] << " "
            << std::setw(6) << std::left << solute.atom(i).name << std::right
            << std::setw(6) << i + 1
            << std::setw(m_width) << vel_rot(0)
            << std::setw(m_width) << vel_rot(1)
            << std::setw(m_width) << vel_rot(2)
            << "\n";
  }

  int index = topo.num_solute_atoms();
  int res_num = 1;

  for (unsigned int s = 0; s < topo.num_solvents(); ++s) {

    for (unsigned int m = 0; m < topo.num_solvent_molecules(s); ++m, ++res_num) {

      for (unsigned int a = 0; a < topo.solvent(s).num_atoms(); ++a, ++index) {
        vel_rot = math::Vec(math::product(Rmat, vel(index)));
        os << std::setw(5) << res_num << " "
                << std::setw(5) << std::left
                << residue_name[topo.solvent(s).atom(a).residue_nr] << " "
                << std::setw(6) << std::left << topo.solvent(s).atom(a).name << std::right
                << std::setw(6) << index + 1
                << std::setw(m_width) << vel_rot(0)
                << std::setw(m_width) << vel_rot(1)
                << std::setw(m_width) << vel_rot(2)
                << "\n";
      }
    }
  }

  os << "END\n";

}

void io::Out_Configuration
::_print_lattice_shifts(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  os << "LATTICESHIFTS\n";
  math::VArray const &shift = conf.special().lattice_shifts;
  for (int i = 0, to = topo.num_atoms(); i < to; ++i) {
    os << std::setw(10) << int(rint(shift(i)(0)))
            << std::setw(10) << int(rint(shift(i)(1)))
            << std::setw(10) << int(rint(shift(i)(2)))
            << "\n";
  }
  os << "END\n";
}

void io::Out_Configuration
::_print_velocityred(configuration::Configuration const &conf,
        int num, std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);
  os << "VELOCITYRED\n";

  math::VArray const &vel = conf.current().vel;
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  math::Vec vel_rot;

  assert(num <= int(vel.size()));
  for (int i = 0; i < num; ++i) {
    vel_rot = math::Vec(math::product(Rmat, vel(i)));
    os << std::setw(m_width) << vel_rot(0)
            << std::setw(m_width) << vel_rot(1)
            << std::setw(m_width) << vel_rot(2)

            << "\n";
    if ((i + 1) % 10 == 0) os << '#' << std::setw(10) << i + 1 << "\n";
  }

  os << "END\n";
}

void io::Out_Configuration
::_print_force(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os, bool constraint_force) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_force_precision);

  os << "FREEFORCE\n";

  math::VArray const & force = conf.current().force;
  topology::Solute const &solute = topo.solute();
  std::vector<std::string> const &residue_name = topo.residue_names();
  //matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  math::Vec force_rot;
  os << "# first 24 chars ignored\n";
  for (int i = 0, to = topo.num_solute_atoms(); i < to; ++i) {
    force_rot = math::Vec(math::product(Rmat, force(i)));
    os << std::setw(6) << solute.atom(i).residue_nr + 1
            << std::setw(5) << residue_name[solute.atom(i).residue_nr]
            << std::setw(6) << solute.atom(i).name
            << std::setw(8) << i + 1
            << std::setw(m_force_width) << force_rot(0)
            << std::setw(m_force_width) << force_rot(1)
            << std::setw(m_force_width) << force_rot(2)
            << "\n";
  }
  int index = topo.num_solute_atoms();

  for (unsigned int s = 0; s < topo.num_solvents(); ++s) {
    for (unsigned int m = 0; m < topo.num_solvent_molecules(s); ++m) {
      for (unsigned int a = 0; a < topo.solvent(s).num_atoms(); ++a, ++index) {
        force_rot = math::Vec(math::product(Rmat, force(index)));
        os << std::setw(6) << topo.solvent(s).atom(a).residue_nr + 1
                << std::setw(5) << residue_name[topo.solvent(s).atom(a).residue_nr]
                << std::setw(6) << topo.solvent(s).atom(a).name
                << std::setw(8) << index + 1
                << std::setw(m_force_width) << force_rot(0)
                << std::setw(m_force_width) << force_rot(1)
                << std::setw(m_force_width) << force_rot(2)
                << "\n";
      }
    }
  }
  os << "END\n";

  if (constraint_force) {
    os << "CONSFORCE\n";

    math::VArray const & cons_force = conf.current().constraint_force;

    os << "# first 24 chars ignored\n";
    for (int i = 0, to = topo.num_solute_atoms(); i < to; ++i) {
      force_rot = math::Vec(math::product(Rmat, cons_force(i)));
      os << std::setw(6) << solute.atom(i).residue_nr + 1
              << std::setw(5) << residue_name[solute.atom(i).residue_nr]
              << std::setw(6) << solute.atom(i).name
              << std::setw(8) << i + 1
              << std::setw(m_force_width) << force_rot(0)
              << std::setw(m_force_width) << force_rot(1)
              << std::setw(m_force_width) << force_rot(2)
              << "\n";
    }
    index = topo.num_solute_atoms();

    for (unsigned int s = 0; s < topo.num_solvents(); ++s) {
      for (unsigned int m = 0; m < topo.num_solvent_molecules(s); ++m) {
        for (unsigned int a = 0; a < topo.solvent(s).num_atoms(); ++a, ++index) {
           force_rot = math::Vec(math::product(Rmat, cons_force(index)));
          os << std::setw(6) << topo.solvent(s).atom(a).residue_nr + 1
                  << std::setw(5) << residue_name[topo.solvent(s).atom(a).residue_nr]
                  << std::setw(6) << topo.solvent(s).atom(a).name
                  << std::setw(8) << index + 1
                  << std::setw(m_force_width) << force_rot(0)
                  << std::setw(m_force_width) << force_rot(1)
                  << std::setw(m_force_width) << force_rot(2)
                  << "\n";
        }
      }
    }
    os << "END\n";
  }
}

void io::Out_Configuration
::_print_forcered(configuration::Configuration const &conf,
        int num,
        std::ostream &os,
        bool constraint_force) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_force_precision);

  os << "FREEFORCERED\n";

  const math::VArray &force = conf.old().force;
//matrix to rotate back into orignial Cartesian Coordinat system
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  math::Vec force_rot;
  
  assert(num <= int(force.size()));

  for (int i = 0; i < num; ++i) {
    force_rot = math::Vec(math::product(Rmat, force(i)));
    os << std::setw(m_force_width) << force_rot(0)
            << std::setw(m_force_width) << force_rot(1)
            << std::setw(m_force_width) << force_rot(2)
            << "\n";

    if ((i + 1) % 10 == 0) os << '#' << std::setw(10) << i + 1 << "\n";
  }

  os << "END\n";

  if (constraint_force) {
    os << "CONSFORCERED\n";

    const math::VArray & cons_force = conf.old().constraint_force;
    assert(num <= int(cons_force.size()));
    math::Vec cons_force_rot;
    for (int i = 0; i < num; ++i) {
      cons_force_rot = math::Vec(math::product(Rmat, cons_force(i)));
      os << std::setw(m_force_width) << cons_force_rot(0)
              << std::setw(m_force_width) << cons_force_rot(1)
              << std::setw(m_force_width) << cons_force_rot(2)
              << "\n";

      if ((i + 1) % 10 == 0) os << '#' << std::setw(10) << i + 1 << "\n";
    }

    os << "END\n";
  }
}

/**
 * 
 * @section energyredhelper ENERGY03
 *
@verbatim
ENERGY03
# totals
   1.598443703e+02 # total
   1.709320077e+02 # kinetic
  -4.205278511e+01 # potential total
   3.980457903e+01 # covalent total
   1.935607149e+01 # bonds total
   1.267041319e+01 # angles total
   1.485470503e+00 # impropers total
   6.292623846e+00 # dihedrals total
   0.000000000e+00 # crossdihedrals total
  -8.185736413e+01 # nonbonded total
  -1.442773701e+00 # Lennard-Jones total
  -8.041459043e+01 # Coulomb/Reaction-Field total
   0.000000000e+00 # lattice total
   0.000000000e+00 # lattice sum pair total
   0.000000000e+00 # lattice sum real space total
   0.000000000e+00 # lattice sum k (reciprocal) space total
   0.000000000e+00 # lattice sum A term total
   0.000000000e+00 # lattice sum self total
   0.000000000e+00 # lattice sum surface total
   0.000000000e+00 # polarisation self total
   0.000000000e+00 # special total
   0.000000000e+00 # SASA total
   0.000000000e+00 # SASA volume total
   0.000000000e+00 # constraints total
   0.000000000e+00 # distance restraints total
   0.000000000e+00 # dihedral restraints total
   0.000000000e+00 # position restraints total
   0.000000000e+00 # J-value restraints total
   0.000000000e+00 # X-ray restraints total
   0.000000000e+00 # Local elevation total
   3.096514777e+01 # EDS: energy of reference state
   0.000000000e+00 # Entropy
# baths
# number of baths
2
#  kinetic total     centre of mass    internal/rotational
   2.579417475e+01   8.149832229e-01   2.497919152e+01 # 1-st bath
   1.451378329e+02   7.403204119e+01   7.110579174e+01 # 2-nd bath
# bonded
# number of energy groups
2
#  bond              angle             improper          dihedral        crossdihedral
   1.935607149e+01   1.267041319e+01   1.485470503e+00   6.292623846e+00 0.0000000e+00  # energy group 1
   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00 0.0000000e+00  # energy group 2
# nonbonded
#  Lennard-Jones     Coulomb/RF        lattice sum real  lattice sum reciproc.
  -4.896399521e-02  -6.781355366e+01   0.000000000e+00   0.000000000e+00  # 1 - 1
  -6.774710271e-01   4.920740888e-01   0.000000000e+00   0.000000000e+00  # 1 - 2
  -7.163386790e-01  -1.309311086e+01   0.000000000e+00   0.000000000e+00  # 2 - 2
# special
#  constraints       pos. restraints   dist. restraints  dihe. restr.      SASA              SASA volume       jvalue            local elevation   path integral
   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00 # group 1
   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00   0.000000000e+00 # group 2
# eds (enveloping distribution sampling)
# numstates
2
           # total         nonbonded          special
   3.096514777e+01   3.096514777e+01   0.000000000e+00
   3.096514777e+01   3.096514777e+01   0.000000000e+00
END
@endverbatim
 *
 */
void io::Out_Configuration
::_print_energyred(configuration::Configuration const &conf,
        std::ostream &os) {
  os.setf(std::ios::scientific, std::ios::floatfield);
  os.precision(m_precision);

  os << "ENERGY03\n";
  _print_energyred_helper(os, conf.old().energies);
  os << "END\n";

}

void io::Out_Configuration
::write_replica_energy(util::Replica_Data const & replica_data,
        simulation::Simulation const & sim,
        configuration::Energy const & energy,
        int reeval) {
  std::ostream &replica_traj = m_replica_traj;
  replica_traj.setf(std::ios::scientific, std::ios::floatfield);
  replica_traj.precision(m_precision);

  print_REMD(replica_traj, replica_data, sim.param(), reeval);
  _print_timestep(sim, replica_traj);

  replica_traj << "ENERGY03\n";
  _print_energyred_helper(replica_traj, energy);
  replica_traj << "END\n";

}

void io::Out_Configuration
::_print_free_energyred(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  // assert(m_free_energy_traj.is_open());

  os.setf(std::ios::scientific, std::ios::floatfield);
  os.precision(m_precision);

  os << "FREEENERDERIVS03\n"
          << "# lambda\n"
          << std::setw(18) << topo.old_lambda() << "\n";

  _print_energyred_helper(os, conf.old().perturbed_energy_derivatives);

  os << "END\n";

}

/**
 * 
 * @section volumepressurered VOLUMEPRESSURE03
 *
@verbatim
VOLUMEPRESSURE03
# mass
   5.044822000e+02
# temperature
# number of temperature coupling baths
2
#  total             com               ir                scaling factor
   1.723525431e+02   6.534704791e+01   1.820803154e+02   1.128031828e+00 # 1st bath
   2.909363241e+02   2.968021431e+02   2.850705051e+02   1.002070048e+00 # 2nd bath
# volume
   5.345718909e+01
# box
   3.767055681e+00   0.000000000e+00   0.000000000e+00 # K
   0.000000000e+00   3.767055681e+00   0.000000000e+00 # L
   0.000000000e+00   0.000000000e+00   3.767055681e+00 # M
# pressure
   7.348382177e-01
   5.261908890e+00
   2.490310167e+01
#  pressure tensor
   3.260036299e-01  -3.892307215e-01   3.337546495e-01
  -4.705767676e-02   8.540193624e-01   1.012276325e-01
   1.604604267e-01   2.852680401e-01   1.024491661e+00
#  virial tensor
   1.508936821e+01   1.501353732e+01   1.232971839e+00
   5.867732745e+00   4.634376722e+00   1.731944968e+00
   5.864882856e+00  -3.187196467e+00  -3.938018259e+00
#  molecular kinetic energy tensor
   2.380298705e+01   4.609947183e+00   1.015376454e+01
   4.609947183e+00   2.746111399e+01   4.437617314e+00
   1.015376454e+01   4.437617314e+00   2.344520396e+01
@endverbatim
 *
 */
void io::Out_Configuration
::_print_volumepressurered(topology::Topology const & topo,
        configuration::Configuration const &conf,
        simulation::Simulation const &sim,
        std::ostream &os) {
  std::vector<double> const s;

  os.setf(std::ios::scientific, std::ios::floatfield);
  os.precision(m_precision);

  os << "VOLUMEPRESSURE03\n";

  _print_volumepressurered_helper(os,
          math::sum(topo.mass()),
          conf.old().phi,conf.old().theta, conf.old().psi,
          sim.multibath(),
          s,
          conf.old().energies,
          conf.old().box,
          conf.boundary_type,
          conf.old().pressure_tensor,
          conf.old().virial_tensor,
          conf.old().kinetic_energy_tensor);

  os << "END\n";

}

template<math::boundary_enum b>
void _print_ramd_bound(topology::Topology const & topo,
        configuration::Configuration const &conf,
        simulation::Simulation const &sim,
        std::ostream &os, int width) {

  math::Periodicity<b> periodicity(conf.current().box);
  math::Vec com(0.0, 0.0, 0.0);
  math::Vec r;
  math::Vec f = conf.special().ramd.force_direction * sim.param().ramd.fc;
  std::set<unsigned int>::const_iterator
  it = sim.param().ramd.atom.begin(),
          i0 = sim.param().ramd.atom.begin(),
          to = sim.param().ramd.atom.end();

  for (; it != to; ++it) {
    periodicity.nearest_image(conf.current().pos(*it),
            conf.current().pos(*i0), r);
    com += topo.mass()(*it) * r;
  }

  com /= conf.special().ramd.total_mass;
  com += conf.current().pos(*i0);
  //rotate to orignial Cartesian coordinates
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  if (conf.boundary_type == math::truncoct)
    Rmat = math::product(Rmat, math::truncoct_triclinic_rotmat(false));
  f = math::Vec(math::product(Rmat, f));
  com = math::Vec(math::product(Rmat, com));
  os << "# force\n";
  os << std::setw(width) << f(0)
          << std::setw(width) << f(1)
          << std::setw(width) << f(2)
          << "\n";
  os << "# com RAMD atoms\n";
  os << std::setw(width) << com(0)
          << std::setw(width) << com(1)
          << std::setw(width) << com(2)
          << "\n";

}

void io::Out_Configuration
::_print_ramd(topology::Topology const &topo,
        configuration::Configuration const &conf,
        simulation::Simulation const &sim,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);
  os << "RAMD\n";

  SPLIT_BOUNDARY(_print_ramd_bound,
          topo, conf, sim, os, m_width);
  os << "END\n";

}

void io::Out_Configuration
::_print_box(configuration::Configuration const &conf,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);
  //change to GENBOX
  os << "GENBOX\n";

  math::Box box = conf.current().box;

  os << std::setw(5) << conf.boundary_type << "\n";

  long double a, b, c, alpha, beta, gamma, phi, theta, psi;
  math::Matrixl Rmat(math::rmat(conf.current().phi,
          conf.current().theta, conf.current().psi));
  
  math::Box m(0.0);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        m(j)(i) += Rmat(k, i) * box(j)(k);
  box = m;

  // convert it back to truncoct
  if (conf.boundary_type == math::truncoct)
    math::truncoct_triclinic_box(box, false);
  
  a = math::abs(box(0));
  b = math::abs(box(1));
  c = math::abs(box(2));

  os << std::setw(m_width) << a
          << std::setw(m_width) << b
          << std::setw(m_width) << c
          << "\n";


  if (a != 0.0 && b != 0.0 && c != 0.0) {
    alpha = acos(math::costest(dot(box(1), box(2)) / (b * c)));
    beta = acos(math::costest(dot(box(0), box(2)) / (a * c)));
    gamma = acos(math::costest(dot(box(0), box(1)) / (a * b)));

    os << std::setw(m_width) << alpha * 180 / math::Pi
            << std::setw(m_width) << beta * 180 / math::Pi
            << std::setw(m_width) << gamma * 180 / math::Pi
            << "\n";

    /* we are already in the frame of the box!
        math::Matrixl Rmat = (math::rmat(box));

        long double R11R21 = sqrtl(Rmat(0, 0) * Rmat(0, 0) + Rmat(0, 1) * Rmat(0, 1));
        if (R11R21 == 0.0) {
          theta = -math::sign(Rmat(0, 2)) * M_PI / 2;
          psi = 0.0;
          phi = -math::sign(Rmat(1, 0)) * acosl(math::costest(Rmat(1, 1)));
        } else {
          theta = -math::sign(Rmat(0, 2)) * acosl(math::costest(R11R21));
          long double costheta = cosl(theta);
          psi = math::sign(Rmat(1, 2) / costheta) * acosl(math::costest(Rmat(2, 2) / costheta));
          phi = math::sign(Rmat(0, 1) / costheta) * acosl(math::costest(Rmat(0, 0) / costheta));

        }

     */
  
    if(fabs(conf.current().phi)<math::epsilon)
      phi=0.0;
    else phi=conf.current().phi;
    if(fabs(conf.current().theta)<math::epsilon)
      theta=0.0;
    else theta=conf.current().theta;
    if(fabs(conf.current().psi)<math::epsilon)
      psi=0.0;
    else psi=conf.current().psi;
    os << std::setw(m_width) << phi * 180 / math::Pi
            << std::setw(m_width) << theta * 180 / math::Pi
            << std::setw(m_width) << psi * 180 / math::Pi
            << "\n";
  } else {
    os << std::setw(m_width) << 0.0
            << std::setw(m_width) << 0.0
            << std::setw(m_width) << 0.0
            << "\n";
    os << std::setw(m_width) << 0.0
            << std::setw(m_width) << 0.0
            << std::setw(m_width) << 0.0
            << "\n";
  }
  double origin = 0.0;

  os << std::setw(m_width) << origin
          << std::setw(m_width) << origin
          << std::setw(m_width) << origin
          << "\n";


  os << "END\n";

}

void io::Out_Configuration
::precision(int prec, int add) {
  m_precision = prec;
  m_width = prec + add;
}

void io::Out_Configuration
::force_precision(int prec, int add) {
  m_force_precision = prec;
  m_force_width = prec + add;
}

int io::Out_Configuration
::precision() {
  return m_precision;
}

int io::Out_Configuration
::force_precision() {
  return m_force_precision;
}

void io::Out_Configuration
::print(topology::Topology const & topo,
        configuration::Configuration & conf,
        simulation::Simulation const & sim) {
  if (sim.param().print.stepblock && (sim.steps() % sim.param().print.stepblock) == 0) {

    m_output << "\n---------------------------------------------------"
            << "-----------------------------\n";

    _print_timestep(sim, m_output);

    print_ENERGY(m_output, conf.old().energies, topo.energy_groups());

    if (sim.param().sasa.switch_sasa)
      print_sasa(m_output, topo, conf, sim, "SOLVENT ACCESSIBLE SURFACE AREAS AND VOLUME");

    if (sim.param().perturbation.perturbation) {

      m_output << "lambda: " << topo.old_lambda() << "\n";

      print_ENERGY(m_output, conf.old().perturbed_energy_derivatives,
              topo.energy_groups(), "dE/dLAMBDA", "dE_");
    }

    print_MULTIBATH(m_output, sim.multibath(), conf.old().energies);

    // flexible shake kinetic energy
    if (sim.param().constraint.solute.algorithm == simulation::constr_flexshake) {
      m_output << "FLEXSHAKE\n";
      m_output << "\tflex_ekin";
      for (unsigned int i = 0; i < conf.special().flexible_constraint.flexible_ekin.size(); ++i)
        m_output << std::setw(12) << std::setprecision(4) << std::scientific
              << conf.special().flexible_constraint.flexible_ekin[i];
      m_output << "\nEND\n";
    }

    if (sim.param().pcouple.calculate)
      print_PRESSURE(m_output, conf);

    m_output.flush();

  }
  if (sim.param().ramd.fc != 0.0 &&
          sim.param().ramd.every &&
          (sim.steps() % sim.param().ramd.every) == 0) {
    print_RAMD(m_output, conf, topo.old_lambda());
  }
}

void io::Out_Configuration
::print_final(topology::Topology const & topo,
        configuration::Configuration & conf,
        simulation::Simulation const & sim) {
  m_output << "\n============================================================\n";
  m_output << "FINAL DATA\n";
  m_output << "============================================================\n\n\n";

  m_output << "\tsimulation time  :" << std::setw(10) << sim.time() << "\n"
          << "\tsimulation steps :" << std::setw(10) << sim.steps() << "\n\n";

  configuration::Energy e, ef;
  math::Matrix p, pf, v, vf, et, etf;

  std::vector<double> sasa_a, sasa_af, sasa_vol, sasa_volf;
  double sasa_tot, sasa_totf, sasa_voltot, sasa_voltotf;

  if (sim.param().minimise.ntem) {
    print_ENERGY(m_output, conf.current().energies, topo.energy_groups(), "MINIMIZED ENERGY",
            "<EMIN>_");
  }

  // new averages
  conf.current().averages.simulation().energy_average(e, ef);
  conf.current().averages.simulation().pressure_average(p, pf, v, vf, et, etf);

  // averages and fluctuation for sasa and volume calculation
  if (sim.param().sasa.switch_sasa) {
    conf.current().averages.simulation().sasa_average(sasa_a, sasa_af, sasa_tot, sasa_totf);
    if (sim.param().sasa.switch_volume)
      conf.current().averages.simulation().sasavol_average(sasa_vol, sasa_volf, sasa_voltot, sasa_voltotf);
  }

  print_ENERGY(m_output, e, topo.energy_groups(), "ENERGY AVERAGES", "<E>_");
  print_ENERGY(m_output, ef, topo.energy_groups(), "ENERGY FLUCTUATIONS", "<<E>>_");
  print_MULTIBATH(m_output, sim.multibath(), e, "TEMPERATURE AVERAGES");
  print_MULTIBATH(m_output, sim.multibath(), ef, "TEMPERATURE FLUCTUATIONS");

  if (sim.param().pcouple.calculate) {
    print_MATRIX(m_output, p, "PRESSURE AVERAGE");
    print_MATRIX(m_output, pf, "PRESSURE FLUCTUATION");
  }

  if (sim.param().perturbation.perturbation) {

    double lambda, lambda_fluct;
    conf.current().averages.simulation().
            energy_derivative_average(e, ef, lambda, lambda_fluct, sim.param().perturbation.dlamt);

    if (sim.param().perturbation.dlamt) {

      print_ENERGY(m_output, e, topo.energy_groups(), "CUMULATIVE DG", "DG_");

      // what's that anyway...
      //print_ENERGY(m_output, ef, topo.energy_groups(), "DG FLUCTUATIONS", "<<DG>>_");
    }
    else {

      std::ostringstream ss, pre;
      ss << "dE/dLAMBDA ";
      pre << "dE/dl";

      print_ENERGY(m_output, e, topo.energy_groups(),
              ss.str() + "AVERAGES", "<" + pre.str() + ">_");


      print_ENERGY(m_output, ef, topo.energy_groups(),
              ss.str() + "FLUCTUATIONS", "<<" + pre.str() + ">>_");

    }
  }
  if (sim.param().ramd.fc != 0.0 && sim.param().ramd.every)
    print_RAMD(m_output, conf, topo.old_lambda());

  // print sasa and volume averages, fluctuations
  if (sim.param().sasa.switch_sasa){
    int volume = sim.param().sasa.switch_volume;
    print_sasa_avg(m_output, sasa_a, sasa_vol, sasa_tot, sasa_voltot, "SASA AND VOLUME AVERAGE", volume);
    print_sasa_fluct(m_output, sasa_af, sasa_volf, sasa_totf, sasa_voltotf, "SASA AND VOLUME FLUCTUATION", volume);

    //print_forces(m_output, topo, conf, sim);
  }

}

void io::Out_Configuration
::_print_blockaveraged_energyred(configuration::Configuration const &conf,
        std::ostream &os) {
  os.setf(std::ios::scientific, std::ios::floatfield);
  os.precision(m_precision);

  configuration::Energy e, ef;
  // energies are in old(), but averages in current()!
  conf.current().averages.block().energy_average(e, ef);

  os << "BAENERGY03\n";
  _print_energyred_helper(os, e);
  os << "END\n";

  os << "BAEFLUCT03\n";
  _print_energyred_helper(os, ef);
  os << "END\n";

}

void io::Out_Configuration
::_print_blockaveraged_volumepressurered(configuration::Configuration const & conf,
        simulation::Simulation const & sim,
        std::ostream &os) {
  double mass, massf, vol, volf;
  std::vector<double> s, sf;
  configuration::Energy e, ef;
  math::Box b, bf;
  math::Matrix p, pf, v, vf, k, kf;

  // needed again. do in 1 function (energyred && volumepressurered)
  // energies in old(), but averages in current()!
  conf.current().averages.block().energy_average(e, ef);
  conf.current().averages.block().pressure_average(p, pf, v, vf, k, kf);
  conf.current().averages.block().mbs_average(mass, massf, vol, volf, b, bf, s, sf);

  os.setf(std::ios::scientific, std::ios::floatfield);
  os.precision(m_precision);

  os << "BAVOLUMEPRESSURE03\n";

  _print_volumepressurered_helper(os,
          mass,
          conf.old().phi,conf.old().theta, conf.old().psi,
          sim.multibath(),
          s,
          e,
          b,
          conf.boundary_type,
          p,
          v,
          k);

  os << "END\n";

  os << "BAVPFLUCT03\n";

  _print_volumepressurered_helper(os,
          massf,
          conf.old().phi,conf.old().theta, conf.old().psi,
          sim.multibath(),
          sf,
          ef,
          bf,
          conf.boundary_type,
          pf,
          vf,
          kf);

  os << "END\n";

}

void io::Out_Configuration
::_print_blockaveraged_free_energyred(configuration::Configuration const &conf,
        double dlamt,
        std::ostream &os) {
  os.setf(std::ios::scientific, std::ios::floatfield);
  os.precision(m_precision);

  configuration::Energy e, ef;
  double lambda, lambda_fluct;

  // energies in old(), but averages in current()!
  conf.current().averages.block().energy_derivative_average(e, ef, lambda, lambda_fluct, dlamt);

  os << "BAFREEENERDERIVS03\n"
          << "# lambda\n"
          << std::setw(18) << lambda << "\n";

  _print_energyred_helper(os, e);

  os << "END\n";

  os << "BAFREEEFLUCTS03\n"
          << "# lambda\n"
          << std::setw(18) << lambda_fluct << "\n";

  _print_energyred_helper(os, ef);

  os << "END\n";

}

void io::Out_Configuration::_print_flexv(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  DEBUG(10, "FLEXV");

  unsigned int k = 0;

  std::vector<topology::two_body_term_struct>::const_iterator
  constr_it = topo.solute().distance_constraints().begin(),
          constr_to = topo.solute().distance_constraints().end();

  os << "FLEXV\n";
  os << "#\tflexible constraints ("
          << topo.solute().distance_constraints().size()
          << ")\n";

  for (; constr_it != constr_to; ++constr_it, ++k) {

    assert(conf.special().flexible_constraint.flex_len.size() > k);
    assert(conf.special().flexible_constraint.flexible_vel.size() > k);

    os << std::setw(15) << constr_it->i + 1
            << std::setw(10) << constr_it->j + 1
            << std::setw(20) << conf.special().flexible_constraint.flex_len[k]
            << std::setw(20) << conf.special().flexible_constraint.flexible_vel[k]
            << "\n";
  }

  std::vector<topology::perturbed_two_body_term_struct>::const_iterator
  pconstr_it = topo.perturbed_solute().distance_constraints().begin(),
          pconstr_to = topo.perturbed_solute().distance_constraints().end();

  os << "#\tperturbed flexible constraints ("
          << topo.perturbed_solute().distance_constraints().size()
          << " of "
          << conf.special().flexible_constraint.flex_len.size()
          << ")\n";

  for (; pconstr_it != pconstr_to; ++pconstr_it, ++k) {

    assert(conf.special().flexible_constraint.flex_len.size() > k);
    assert(conf.special().flexible_constraint.flexible_vel.size() > k);

    os << std::setw(15) << pconstr_it->i + 1
            << std::setw(10) << pconstr_it->j + 1
            << std::setw(20) << conf.special().flexible_constraint.flex_len[k]
            << std::setw(20) << conf.special().flexible_constraint.flexible_vel[k]
            << "\n";
  }

  os << "END\n";

}

void io::Out_Configuration::_print_stochastic_integral(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  topology::Solute const &solute = topo.solute();
  std::vector<std::string> const &residue_name = topo.residue_names();

  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision - 2); // because it's scientific

  os << "STOCHINT\n";
  os << "# first 24 chars ignored\n";

  for (int i = 0, to = topo.num_solute_atoms(); i < to; ++i) {
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << std::setw(5) << solute.atom(i).residue_nr + 1 << " "
            << std::setw(5) << std::left
            << residue_name[solute.atom(i).residue_nr] << " "
            << std::setw(6) << std::left << solute.atom(i).name << std::right
            << std::setw(6) << i + 1;
    os.setf(std::ios::scientific, std::ios::floatfield);
    os << std::setw(m_width) << conf.current().stochastic_integral(i)(0)
            << std::setw(m_width) << conf.current().stochastic_integral(i)(1)
            << std::setw(m_width) << conf.current().stochastic_integral(i)(2)
            << "\n";
  }

  int index = topo.num_solute_atoms();
  int res_nr = 1;

  for (unsigned int s = 0; s < topo.num_solvents(); ++s) {

    for (unsigned int m = 0; m < topo.num_solvent_molecules(s); ++m, ++res_nr) {

      for (unsigned int a = 0; a < topo.solvent(s).num_atoms(); ++a, ++index) {
        os.setf(std::ios::fixed, std::ios::floatfield);
        os << std::setw(5) << res_nr
                << ' ' << std::setw(5) << std::left
                << residue_name[topo.solvent(s).atom(a).residue_nr] << " "
                << std::setw(6) << std::left
                << topo.solvent(s).atom(a).name << std::right
                << std::setw(6) << index + 1;
        os.setf(std::ios::scientific, std::ios::floatfield);
        os << std::setw(m_width) << conf.current().stochastic_integral(index)(0)
                << std::setw(m_width) << conf.current().stochastic_integral(index)(1)
                << std::setw(m_width) << conf.current().stochastic_integral(index)(2)
                << "\n";
      }
    }
  }
  os.setf(std::ios::fixed, std::ios::floatfield);
  os << "# seed\n" << std::setw(10) << std::right
          << conf.current().stochastic_seed << "\n";
  os << "END\n";
}

void io::Out_Configuration::_print_pertdata(topology::Topology const &topo,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(7);
  os << "PERTDATA" << std::endl
          << std::setw(15) << topo.lambda() << std::endl
          << "END" << std::endl;
}

void io::Out_Configuration::write_replica_step
(
        simulation::Simulation const & sim,
        util::Replica_Data const & replica_data,
        output_format const form
        ) {
  DEBUG(10, "REPLICA");

  // print to all trajectories
  if (form == reduced) {

    if (m_every_pos && (sim.steps() % m_every_pos) == 0)
      print_REMD(m_pos_traj, replica_data, sim.param());

    if (m_every_vel && (sim.steps() % m_every_vel) == 0)
      print_REMD(m_vel_traj, replica_data, sim.param());

    if (m_every_force && ((sim.steps()) % m_every_force) == 0) {
      if (sim.steps())
        print_REMD(m_force_traj, replica_data, sim.param());
    }

    if (m_every_energy && (sim.steps() % m_every_energy) == 0) {
      if (sim.steps())
        print_REMD(m_energy_traj, replica_data, sim.param());
    }

    if (m_every_free_energy && (sim.steps() % m_every_free_energy) == 0) {
      if (sim.steps())
        print_REMD(m_free_energy_traj, replica_data, sim.param());
    }

    if (m_every_blockaverage && (sim.steps() % m_every_blockaverage) == 0) {
      if (m_write_blockaverage_energy) {
        if (sim.steps())
          print_REMD(m_blockaveraged_energy, replica_data, sim.param());
      }

      if (m_write_blockaverage_free_energy) {
        if (sim.steps())
          print_REMD(m_blockaveraged_free_energy, replica_data, sim.param());
      }
    }
  }// reduced

  else if (form == final && m_final) {
    print_REMD(m_final_conf, replica_data, sim.param());

    // forces and energies still go to their trajectories
    if (m_every_force && ((sim.steps()) % m_every_force) == 0)
      if (sim.steps())
        print_REMD(m_force_traj, replica_data, sim.param());

    if (m_every_energy && (sim.steps() % m_every_energy) == 0)
      print_REMD(m_energy_traj, replica_data, sim.param());

    if (m_every_free_energy && (sim.steps() % m_every_free_energy) == 0)
      print_REMD(m_free_energy_traj, replica_data, sim.param());
  }// final

  else {

    // not reduced or final (so: decorated)

    if (m_every_pos && (sim.steps() % m_every_pos) == 0)
      print_REMD(m_pos_traj, replica_data, sim.param());

    if (m_every_vel && (sim.steps() % m_every_vel) == 0)
      print_REMD(m_vel_traj, replica_data, sim.param());

    if (m_every_force && (sim.steps() % m_every_force) == 0) {
      if (sim.steps())
        print_REMD(m_force_traj, replica_data, sim.param());
    }
  } // decorated

}

void io::Out_Configuration::_print_jvalue(simulation::Parameter const & param,
        configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os, bool formated) {
  DEBUG(10, "JVALUE Averages and LE data");

  if (param.jvalue.mode != simulation::jvalue_restr_inst &&
      param.jvalue.mode != simulation::jvalue_restr_inst_weighted) {
    os << "JVALUERESEXPAVE\n";
    if (formated) {
      os << "# I J K L <J>\n";
    }
    os.setf(std::ios::fixed, std::ios::floatfield);
    os.precision(7);
    std::vector<double>::const_iterator av_it = conf.special().jvalue_av.begin(),
            av_to = conf.special().jvalue_av.end();
    std::vector<topology::jvalue_restraint_struct>::const_iterator jv_it =
            topo.jvalue_restraints().begin();
    for (; av_it != av_to; ++av_it, ++jv_it) {
      if (formated) {
        os << std::setw(5) << jv_it->i+1
           << std::setw(5) << jv_it->j+1
           << std::setw(5) << jv_it->k+1
           << std::setw(5) << jv_it->l+1;
      }
      os << std::setw(15) << *av_it << "\n";
    }
    os << "END\n";
  }

  if (param.jvalue.le && param.jvalue.mode != simulation::jvalue_restr_off) {
    os << "JVALUERESEPS\n";
    if (formated) {
      os << "# I J K L\n# GRID[1.." << param.jvalue.ngrid << "]\n";
    }
    os.setf(std::ios::scientific, std::ios::floatfield);
    os.precision(7);
    std::vector<std::vector<double> >::const_iterator
    le_it = conf.special().jvalue_epsilon.begin(),
            le_to = conf.special().jvalue_epsilon.end();
    std::vector<topology::jvalue_restraint_struct>::const_iterator jv_it =
            topo.jvalue_restraints().begin();

    for (; le_it != le_to; ++le_it, ++jv_it) {
      if (formated) {
        os << std::setw(5) << jv_it->i+1
           << std::setw(5) << jv_it->j+1
           << std::setw(5) << jv_it->k+1
           << std::setw(5) << jv_it->l+1 << "\n";
      }
      for (unsigned int i = 0; i < le_it->size(); ++i)
        os << std::setw(15) << (*le_it)[i];
      os << "\n";
    }
    os << "END\n";
  }
}

void io::Out_Configuration::_print_xray(simulation::Parameter const & param,
        configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os, bool final) {
  DEBUG(10, "XRAY Averages");

  if (param.xrayrest.xrayrest != simulation::xrayrest_inst) {
    os << "XRAYRESEXPAVE\n";
    os.setf(std::ios::fixed, std::ios::floatfield);
    os.precision(7);
    for (unsigned int i=0; i<conf.special().xray_rest.size(); ++i) {
      os << std::setw(15) << conf.special().xray_rest[i].sf_av
         << std::setw(15) << conf.special().xray_rest[i].phase_av << "\n";
    }
    os << "END\n";
  }
}

void io::Out_Configuration::_print_xray_rvalue(simulation::Parameter const & param,
        configuration::Configuration const &conf,
        std::ostream &os) {
  DEBUG(10, "XRAY scaling constants and R-values");

  double k_inst = conf.special().xray.k_inst;
  double k_free_inst = conf.special().xray.k_free_inst;
  double k_avg  = conf.special().xray.k_avg;
  double k_free_avg  = conf.special().xray.k_free_avg;
  double R_inst = conf.special().xray.R_inst;
  double R_avg  = conf.special().xray.R_avg;
  double R_free_inst = conf.special().xray.R_free_inst;
  double R_free_avg  = conf.special().xray.R_free_avg;

  // make sure no rubbish is written
  switch(param.xrayrest.xrayrest) {
    case simulation::xrayrest_off : return;
    case simulation::xrayrest_inst :
      k_avg = k_free_avg = R_avg = R_free_avg = 0.0; break;
    case simulation::xrayrest_avg :
      k_inst = k_free_inst = R_inst = R_free_inst = 0.0; break;
    default: ;// value are OK. do nothing
  }

  os << "XRAYRVALUE\n";
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);
  os << std::setw(15) << k_inst << std::endl
          << std::setw(15) << R_inst << std::endl
          << std::setw(15) << k_free_inst << std::endl
          << std::setw(15) << R_free_inst << std::endl
          << std::setw(15) << k_avg << std::endl
          << std::setw(15) << R_avg << std::endl
          << std::setw(15) << k_free_avg << std::endl
          << std::setw(15) << R_free_avg << std::endl;
  os << "END\n";
}

void io::Out_Configuration::_print_xray_umbrellaweightthresholds(simulation::Parameter const & param,
        topology::Topology const & topo,
        std::ostream &os) {
  DEBUG(10, "XRAY umbrella weight thresholds");

  if (param.xrayrest.local_elevation == false) return;

  os << "XRAYUMBRELLAWEIGHTTHRESHOLDS\n";
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);
  std::vector<topology::xray_umbrella_weight_struct>::const_iterator it =
            topo.xray_umbrella_weights().begin(), to =
            topo.xray_umbrella_weights().end();
  for (; it != to; ++it) {
    os << std::setw(15) << it->threshold
            << std::setw(15) << it->threshold_growth_rate
            << std::setw(15) << it->threshold_overshoot
            << std::setw(3) << (it->threshold_freeze ? 1 : 0) << std::endl;
  }
  os << "END\n";
}

void io::Out_Configuration::_print_distance_restraints(
        configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  DEBUG(10, "distance restraints");

  std::vector<double>::const_iterator av_it = conf.special().distanceres.av.begin(),
          av_to = conf.special().distanceres.av.end();
  std::vector<double>::const_iterator ene_it = conf.special().distanceres.energy.begin();
  std::vector<double>::const_iterator d_it = conf.special().distanceres.d.begin();

  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_distance_restraint_precision);

  os << "DISRESDATA" << std::endl;
  int i;
  for (i = 1; av_it != av_to; ++av_it, ++ene_it, ++d_it, ++i) {
    os << std::setw(m_width) << *d_it
       << std::setw(m_width) << *ene_it;
       if (*av_it != 0) {
         os << std::setw(m_width) << pow(*av_it, -1.0 / 3.0);
       } else {
         os << std::setw(m_width) << 0.0;
       }
       os << std::endl;
  }

  os << "END" << std::endl;
}

void io::Out_Configuration::_print_distance_restraint_averages(
        configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  DEBUG(10, "distance restraint averages");

  std::vector<double>::const_iterator it = conf.special().distanceres.av.begin(),
          to = conf.special().distanceres.av.end();

  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_distance_restraint_precision);

  os << "DISRESEXPAVE" << std::endl;
  int i;
  for (i = 1; it != to; ++it, ++i) {
    os << std::setw(m_width) << *it;
    if (i % 5 == 0)
      os << std::endl;
  }
  if ((i-1) % 5 != 0)
    os << std::endl;

  os << "END" << std::endl;
}

void io::Out_Configuration::_print_dihangle_trans(
        configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  DEBUG(10, "dihedral angle transitions");

  std::vector<double>::const_iterator it = conf.special().dihangle_trans.dihedral_angle_minimum.begin(),
          to = conf.special().dihangle_trans.dihedral_angle_minimum.end();

  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(1);
  unsigned int atom_i, atom_j, atom_k, atom_l;

  os << "D-A-T" << std::endl;
  os << "# Dih. No.    Resid  Atoms                                        Old min. -> New min." << std::endl;
  int i;
  for (i = 1; it != to; ++it, ++i) {
    if (conf.special().dihangle_trans.old_minimum[i] > math::epsilon) {
      atom_i = conf.special().dihangle_trans.i[i];
      atom_j = conf.special().dihangle_trans.j[i];
      atom_k = conf.special().dihangle_trans.k[i];
      atom_l = conf.special().dihangle_trans.l[i];
      os << i << std::setw(14) << conf.special().dihangle_trans.resid[i] + 1
         << std::setw(4) << topo.residue_names()[topo.solute().atom(atom_i).residue_nr]
         << std::setw(4) << topo.solute().atom(atom_i).name << " - "
         << topo.solute().atom(atom_j).name << " - "
         << topo.solute().atom(atom_k).name << " - " << topo.solute().atom(atom_l).name
         << std::setw(4) << atom_i + 1 << " - " << atom_j + 1 << " - "
         << atom_k + 1 << " - " << atom_l + 1
         << std::setw(m_width) << 180.0 * conf.special().dihangle_trans.old_minimum[i] / math::Pi << " -> "
         << 180.0 * conf.special().dihangle_trans.dihedral_angle_minimum[i] / math::Pi
         << std::endl;
    }
  }

  os << "END" << std::endl;
}

void io::Out_Configuration::_print_pscale_jrest(configuration::Configuration const &conf,
        topology::Topology const &topo,
        std::ostream &os) {
  DEBUG(10, "JVALUEPERSCALE data");

  std::vector<topology::jvalue_restraint_struct>::const_iterator
  jval_it = topo.jvalue_restraints().begin(),
          jval_to = topo.jvalue_restraints().end();

  os << "JVALUEPERSCALE\n";
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  for (int i = 0; jval_it != jval_to; ++jval_it, ++i) {
    os << std::setw(10) << conf.special().pscale.scaling[i]
            << std::setw(15) << conf.special().pscale.t[i]
            << "\n";
  }
  os << "END\n";
}

void io::Out_Configuration::_print_replica_information
(
        std::vector<util::Replica_Data> const & replica_data,
        std::ostream &os
        ) {
  DEBUG(10, "replica information");

  std::vector<util::Replica_Data>::const_iterator
  it = replica_data.begin(),
          to = replica_data.end();

  os << "REPDATA\n";

  for (int i = 0; it != to; ++it, ++i) {

    os << std::setw(6) << it->ID
            << " "
            << std::setw(6) << it->run
            << std::setw(6) << it->Ti
            << std::setw(6) << it->li
            << std::setw(18) << it->epot_i
            << std::setw(6) << it->Tj
            << std::setw(6) << it->lj
            << std::setw(18) << it->epot_j
            << std::setw(18) << it->probability
            << std::setw(4) << it->switched
            << std::setw(18) << it->time
            << std::setw(4) << it->state
            << "\n";
  }
  os << "END\n";
}

static void _print_energyred_helper(std::ostream & os, configuration::Energy const &e) {

  const int numenergygroups = unsigned(e.bond_energy.size());
  const int numbaths = unsigned(e.kinetic_energy.size());
  // const int energy_group_size = numenergygroups * (numenergygroups + 1) /2;

  os << "# totals\n";

  os << std::setw(18) << e.total << "\n"// 1
          << std::setw(18) << e.kinetic_total << "\n" // 2
          << std::setw(18) << e.potential_total << "\n" // 3
          << std::setw(18) << e.bonded_total << "\n" // 4
          << std::setw(18) << e.bond_total << "\n" // 5
          << std::setw(18) << e.angle_total << "\n" // 6
          << std::setw(18) << e.improper_total << "\n" // 7
          << std::setw(18) << e.dihedral_total << "\n" // 8
          << std::setw(18) << e.crossdihedral_total << "\n" // 9
          << std::setw(18) << e.nonbonded_total << "\n" // 10
          << std::setw(18) << e.lj_total << "\n" // 11
          << std::setw(18) << e.crf_total << "\n" // 12
          << std::setw(18) << e.ls_total << "\n" // 13
          << std::setw(18) << e.ls_pair_total << "\n" // 14
          << std::setw(18) << e.ls_realspace_total << "\n" // 15
          << std::setw(18) << e.ls_kspace_total << "\n" // 16
          << std::setw(18) << e.ls_a_term_total << "\n" // 17
          << std::setw(18) << e.ls_self_total << "\n" // 18
          << std::setw(18) << e.ls_surface_total << "\n" // 19
          << std::setw(18) << e.self_total << "\n" // 20
          << std::setw(18) << e.special_total << "\n" // 21
          << std::setw(18) << e.sasa_total << "\n" // 22
          << std::setw(18) << e.sasa_volume_total << "\n" // 23
          << std::setw(18) << e.constraints_total << "\n" // 24
          << std::setw(18) << e.distanceres_total << "\n" // 25
          << std::setw(18) << e.dihrest_total << "\n" // 26
          << std::setw(18) << e.posrest_total << "\n" // 27
          << std::setw(18) << e.jvalue_total << "\n" // 28
          << std::setw(18) << e.xray_total << "\n" // 29
          << std::setw(18) << e.leus_total << "\n" // 30
          << std::setw(18) << e.eds_vr << "\n" // 31
          << std::setw(18) << e.entropy_term << "\n"; // 32

  // put eds V_R energy here

  os << "# baths\n";
  os << numbaths << "\n";

  for (int i = 0; i < numbaths; ++i) {
    os << std::setw(18) << e.kinetic_energy[i]
            << std::setw(18) << e.com_kinetic_energy[i]
            << std::setw(18) << e.ir_kinetic_energy[i] << "\n";
  }

  os << "# bonded\n";
  os << numenergygroups << "\n";
  for (int i = 0; i < numenergygroups; i++) {
    os << std::setw(18) << e.bond_energy[i]
            << std::setw(18) << e.angle_energy[i]
            << std::setw(18) << e.improper_energy[i]
            << std::setw(18) << e.dihedral_energy[i]
            << std::setw(18) << e.crossdihedral_energy[i] << "\n";
  }

  os << "# nonbonded\n";
  for (int i = 0; i < numenergygroups; i++) {
    for (int j = i; j < numenergygroups; j++) {
      os << std::setw(18) << e.lj_energy[j][i]
              << std::setw(18) << e.crf_energy[j][i]
              << std::setw(18) << e.ls_real_energy[j][i]
              << std::setw(18) << e.ls_k_energy[j][i] << "\n";
    }
  }

  os << "# special\n";
  for (int i = 0; i < numenergygroups; i++) {
    os << std::setw(18) << e.constraints_energy[i]
            << std::setw(18) << e.posrest_energy[i]
            << std::setw(18) << e.distanceres_energy[i] // disres
            << std::setw(18) << e.dihrest_energy[i] // dihedral res
            << std::setw(18) << e.sasa_energy[i]
            << std::setw(18) << e.sasa_volume_energy[i]
            << std::setw(18) << 0.0 // jval
            << std::setw(18) << 0.0 // local elevation
            << std::setw(18) << 0.0 << "\n"; // path integral
  }

  // eds energy of end states
  os << "# eds\n";
  os << "# numstates\n";
  const unsigned int numstates = e.eds_vi.size();
  os << numstates << "\n";
  os << std::setw(18) << "# total"
          << std::setw(18) << "nonbonded"
          << std::setw(18) << "special\n";
  for (unsigned i = 0; i < e.eds_vi.size(); i++) {
    os << std::setw(18) << e.eds_vi[i]
            << std::setw(18) << e.eds_vi[i] - e.eds_vi_special[i]
            << std::setw(18) << e.eds_vi_special[i] << "\n";
  }

  // write eds energies (vr,{V_i}) here
}

static void _print_volumepressurered_helper(std::ostream &os,
        double mass,
        double const & phi,
        double const & theta,
        double const & psi,
        simulation::Multibath const & m,
        std::vector<double> const & s,
        configuration::Energy const & e,
        math::Box const & b,
        math::boundary_enum t,
        math::Matrix const & p,
        math::Matrix const & v,
        math::Matrix const & k) {
  const int numbaths = int(m.size());

  os << "# mass\n";
  os << std::setw(18) << mass << "\n";

  os << "# temperature\n";
  os << numbaths << "\n";

  for (int i = 0; i < numbaths; ++i) {
    if (m[i].dof)
      os << std::setw(18) << 2 * e.kinetic_energy[i] / math::k_Boltzmann / m[i].dof;
    else
      os << std::setw(18) << 0.0;
    if (m[i].com_dof)
      os << std::setw(18) << 2 * e.com_kinetic_energy[i] / math::k_Boltzmann / m[i].com_dof;
    else
      os << std::setw(18) << 0.0;
    if (m[i].ir_dof)
      os << std::setw(18) << 2 * e.ir_kinetic_energy[i] / math::k_Boltzmann / m[i].ir_dof;
    else
      os << std::setw(18) << 0.0;

    if (s.size())
      os << std::setw(18) << s[i] << "\n";
    else
      os << std::setw(18) << m[i].scale << "\n";

  }

  os << "# volume\n";
  os << std::setw(18) << math::volume(b, t) << "\n";
  //rotate the volume and pressure tensors into the original Cartesian Coordinate system
  math::Matrixl Rmat(math::rmat(phi, theta, psi));
  math::Box b2(math::product(Rmat, b));
  os << std::setw(18) << b2(0)(0) << std::setw(18) << b2(0)(1) << std::setw(18) << b2(0)(2) << "\n"
          << std::setw(18) << b2(1)(0) << std::setw(18) << b2(1)(1) << std::setw(18) << b2(1)(2) << "\n"
          << std::setw(18) << b2(2)(0) << std::setw(18) << b2(2)(1) << std::setw(18) << b2(2)(2) << "\n";

  os << "# pressure\n";
  math::Matrixl auxp(math::product(Rmat, p));
  os << std::setw(18) << (auxp(0, 0) + auxp(1, 1) + auxp(2, 2)) / 3.0 << "\n";
  math::Matrixl auxv(math::product(Rmat, v));
  os << std::setw(18) << (auxv(0, 0) + auxv(1, 1) + auxv(2, 2)) / 3.0 << "\n";
  math::Matrixl auxk(math::product(Rmat, k));
  os << std::setw(18) << (auxk(0, 0) + auxk(1, 1) + auxk(2, 2)) / 3.0 << "\n";

  os << std::setw(18) << auxp(0, 0) << std::setw(18) << auxp(0, 1) << std::setw(18) << auxp(0, 2) << "\n"
          << std::setw(18) << auxp(1, 0) << std::setw(18) << auxp(1, 1) << std::setw(18) << auxp(1, 2) << "\n"
          << std::setw(18) << auxp(2, 0) << std::setw(18) << auxp(2, 1) << std::setw(18) << auxp(2, 2) << "\n";

  os << std::setw(18) << auxv(0, 0) << std::setw(18) << auxv(0, 1) << std::setw(18) << auxv(0, 2) << "\n"
          << std::setw(18) << auxv(1, 0) << std::setw(18) << auxv(1, 1) << std::setw(18) << auxv(1, 2) << "\n"
          << std::setw(18) << auxv(2, 0) << std::setw(18) << auxv(2, 1) << std::setw(18) << auxv(2, 2) << "\n";

  os << std::setw(18) << k(0, 0) << std::setw(18) << k(0, 1) << std::setw(18) << k(0, 2) << "\n"
          << std::setw(18) << k(1, 0) << std::setw(18) << k(1, 1) << std::setw(18) << k(1, 2) << "\n"
          << std::setw(18) << k(2, 0) << std::setw(18) << k(2, 1) << std::setw(18) << k(2, 2) << "\n";

}

void io::Out_Configuration
::_print_position_restraints(simulation::Simulation const & sim,
        topology::Topology const &topo,
        configuration::Configuration const &conf,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "REFPOSITION\n";

  topology::Solute const &solute = topo.solute();
  std::vector<std::string> const &residue_name = topo.residue_names();
  topology::Solvent const &solvent = topo.solvent(0);

  const math::VArray & ref = conf.special().reference_positions;
  const math::SArray & b = conf.special().bfactors;

  for (unsigned int i = 0; i < topo.num_atoms(); ++i) {
    if (i < topo.num_solute_atoms()) {
      os << std::setw(5) << solute.atom(i).residue_nr + 1 << " "
              << std::setw(5) << std::left << residue_name[solute.atom(i).residue_nr] << " "
              << std::setw(6) << std::left << solute.atom(i).name << std::right
              << std::setw(6) << i + 1
              << std::setw(m_width) << ref(i)(0)
              << std::setw(m_width) << ref(i)(1)
              << std::setw(m_width) << ref(i)(2)
              << "\n";
    } else { // just writing out dummy values for first 17 chars
      os << std::setw(5) << "0" << " "
              << std::setw(5) << std::left
              << "SOLV" << " "
              << std::setw(6) << std::left << solvent.atom((i - topo.num_solute_atoms())
              % solvent.atoms().size()).name << std::right
              << std::setw(6) << i + 1
              << std::setw(m_width) << ref(i)(0)
              << std::setw(m_width) << ref(i)(1)
              << std::setw(m_width) << ref(i)(2)
              << "\n";
    }
  }

  os << "END\n";

  if (sim.param().posrest.posrest == simulation::posrest_bfactor) {
    os << "BFACTOR\n";

    for (unsigned int i = 0; i < topo.num_atoms(); ++i) {
      if (i < topo.num_solute_atoms()) {
        os << std::setw(5) << solute.atom(i).residue_nr + 1 << " "
                << std::setw(5) << std::left << residue_name[solute.atom(i).residue_nr] << " "
                << std::setw(6) << std::left << solute.atom(i).name << std::right
                << std::setw(6) << i + 1
                << std::setw(m_width) << b(i)
                << "\n";
      } else {
        os << std::setw(5) << "0" << " "
                << std::setw(5) << std::left
                << "SOLV" << " "
                << std::setw(6) << std::left << solvent.atom((i - topo.num_solute_atoms())
              % solvent.atoms().size()).name << std::right
                << std::setw(6) << i + 1
                << std::setw(m_width) << b(i)
                << "\n";
      }
    }

    os << "END\n";
  }

}

void io::Out_Configuration::
_print_nose_hoover_chain_variables(const simulation::Multibath & multibath,
        std::ostream & os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "NHCVARIABLES\n";
  for (simulation::Multibath::const_iterator it = multibath.begin(), to = multibath.end();
          it != to; ++it) {
    const std::vector<double> & zeta = it->zeta;
    for (std::vector<double>::const_iterator z_it = zeta.begin(), z_to = zeta.end();
            z_it != z_to; ++z_it) {
      os << std::setw(m_width) << *z_it;
    }
    os << "\n";
  }
  os << "END\n";
}

void io::Out_Configuration::
_print_rottrans(configuration::Configuration const &conf,
        simulation::Simulation const &sim,
        std::ostream &os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);

  os << "ROTTRANSREFPOS\n";
  os << "# translational matrix\n";
  for (unsigned int i = 0; i < 3; ++i) {
    for (unsigned int j = 0; j < 3; ++j) {
      os << std::setw(m_width) << conf.special().rottrans_constr.theta_inv_trans(i, j);
    }
    os << "\n";
  }
  os << "# rotational matrix\n";
  for (unsigned int i = 0; i < 3; ++i) {
    for (unsigned int j = 0; j < 3; ++j) {
      os << std::setw(m_width) << conf.special().rottrans_constr.theta_inv_rot(i, j);
    }
    os << "\n";
  }
  os << "# reference positions\n";
  unsigned int last = sim.param().rottrans.last;
  const math::VArray & refpos = conf.special().rottrans_constr.pos;
  for (unsigned int i = 0; i < last; ++i) {
    os << std::setw(m_width) << refpos(i)(0)
            << std::setw(m_width) << refpos(i)(1)
            << std::setw(m_width) << refpos(i)(2) << "\n";
  }
  os << "END\n";
}

void io::Out_Configuration::
_print_umbrellas(configuration::Configuration const & conf, std::ostream & os) {
  os.setf(std::ios::fixed, std::ios::floatfield);
  os.precision(m_precision);
  const std::vector<util::Umbrella> & umb = conf.special().umbrellas;
  os << "LEUSBIAS\n# NUMUMBRELLAS\n"
     << umb.size()
     << "\n";
  for(unsigned int i = 0; i < umb.size(); ++i) {
    os << "# NLEPID NDIM CLES\n";
    os << std::setw(10) << umb[i].id << std::setw(10) << umb[i].dim() << std::setw(15)
            << umb[i].force_constant << "\n";
    os << "# VARTYPE(N) NTLEFU(N) WLES(N) RLES(N) NGRID(N) GRIDMIN(N) GRIDMAX(N)\n";
    for(unsigned int j = 0; j < umb[i].dim(); ++j) {
      os << std::setw(10) << umb[i].variable_type[j]
              << std::setw(10) << umb[i].functional_form[j]
              << std::setw(15) << umb[i].width[j]
              << std::setw(15) << umb[i].cutoff[j]
              << std::setw(10) << umb[i].num_grid_points[j]
              << std::setw(15) << umb[i].grid_min[j]
              << std::setw(15) << umb[i].grid_max[j] << "\n";
    }
    os << "# NCONLE\n";
    os << std::setw(10) << umb[i].configurations.size() << "\n";
    os << "# NVISLE ICONF(1..NDIM)\n";
    std::map<util::Umbrella::leus_conf,util::Umbrella_Weight*>::const_iterator conf_it =
    umb[i].configurations.begin(), conf_to = umb[i].configurations.end();
    for(; conf_it != conf_to; ++conf_it) {
      os << *(conf_it->second);
      for(unsigned int j = 0; j < umb[i].dim(); j++) {
        // here we have to convert to fortran and add 1 because the grid
        // goes form 1 to NDIM in the output format
        os << std::setw(10) << conf_it->first.pos[j]+1;
      }
      os << "\n";
    }
  }
  os << "END\n";
}

/**
 * @file read_special.cc
 * implementation of function read_special
 */

#include "../stdheader.h"
#include <fstream>

#include "../algorithm/algorithm.h"
#include "../topology/topology.h"
#include "../simulation/simulation.h"
#include "../configuration/configuration.h"
#include "../interaction/interaction.h"
#include "../interaction/interaction_types.h"

#include "../io/argument.h"
#include "../io/blockinput.h"
#include "../io/instream.h"
#include "../io/configuration/inframe.h"
#include "../io/configuration/in_configuration.h"
#include "../io/topology/in_topology.h"
#include "../io/topology/in_perturbation.h"
#include "../io/parameter/in_parameter.h"
#include "../io/topology/in_posres.h"
#include "../io/topology/in_distanceres.h"
#include "../io/topology/in_dihrest.h"
#include "../io/topology/in_jvalue.h"
#include "../io/topology/in_friction.h"
#include "../io/topology/in_xray.h"
#include "../io/topology/in_leus.h"
#include "../io/topology/in_qmmm.h"
#include "../util/coding.h"

#include "read_special.h"


#undef MODULE
#undef SUBMODULE
#define MODULE simulation
#define SUBMODULE read_input

/**
 * WARNING: configuration has not been read yet
 */
int io::read_special(io::Argument const & args,
		     topology::Topology & topo,
		     configuration::Configuration & conf,
		     simulation::Simulation & sim,
		     std::ostream & os,
		     bool quiet)
{
  // POSRES
  if (sim.param().posrest.posrest){
    io::igzstream posresspec_file;
  
    if (args.count("posresspec") != 1){
      io::messages.add("position restraints: no data file specified (use @posresspec)",
		       "read special", io::message::error);
    } else {
      posresspec_file.open(args["posresspec"].c_str());
      if (!posresspec_file.is_open()){
	io::messages.add("opening posresspec file failed!\n",
			 "read_special", 
			 io::message::error);
      } else {
        io::In_Posresspec ip(posresspec_file);
        ip.quiet = quiet;

        ip.read(topo, sim, os);
        io::messages.add("position restraints specifciation read from " +
                args["posresspec"] + "\n" + util::frame_text(ip.title),
                "read special", io::message::notice);
      }
    }
    
    io::igzstream refpos_file;

    // check whether we also need the position restraints file containing the
    // positions and B-factors
    if (sim.param().posrest.posrest == simulation::posrest_bfactor ||
        sim.param().posrest.read) {

      if (args.count("refpos") != 1) {
        io::messages.add("position restraints: no data file specified (use @refpos)",
                "read special", io::message::error);
      } else {
        refpos_file.open(args["refpos"].c_str());
        if (!refpos_file.is_open()) {
          io::messages.add("opening refpos file failed!\n",
                  "read_special",
                  io::message::error);
        } else {
          io::In_Refpos ip(refpos_file);
          ip.quiet = quiet;

          ip.read(topo, conf, sim, os);
          io::messages.add("position restraints specifciation read from " +
                  args["refpos"] + "\n" + util::frame_text(ip.title),
                  "read special", io::message::notice);
        }
      }
    }
  } // POSRES

  // DISTANCERES
  if (sim.param().distanceres.distanceres){
    io::igzstream distanceres_file;

    if (args.count("distrest") != 1){
      io::messages.add("distance restraints: no data file specified (use @distrest)",
		       "read special", io::message::error);
    } else {
      distanceres_file.open(args["distrest"].c_str());
      if (!distanceres_file.is_open()){
	io::messages.add("opening distanceres file failed!\n",
			 "read_special", io::message::error);
      } else {
        io::In_Distanceres ip(distanceres_file);
        ip.quiet = quiet;

        ip.read(topo, sim, os);
        io::messages.add("distance restraints read from " + args["distrest"] +
                "\n" + util::frame_text(ip.title),
                "read special", io::message::notice);
      }
    }    
  } // DISTANCERES

  // DIHREST
  if (sim.param().dihrest.dihrest != simulation::dihedral_restr_off){
    io::igzstream dihrest_file;

    if (args.count("dihrest") != 1){
      io::messages.add("dihedral restraints: no data file specified (use @dihrest)",
		       "read special", io::message::error);
    } else{
      dihrest_file.open(args["dihrest"].c_str());
      if (!dihrest_file.is_open()){
	io::messages.add("opening dihrest file '" + args["dihrest"] + "'failed!\n",
			 "read_special", io::message::error);
      } else {
        io::In_Dihrest ip(dihrest_file);
        ip.quiet = quiet;

        ip.read(topo, sim, os);
        io::messages.add("dihedral restraints read from " + args["dihrest"] +
                "\n" + util::frame_text(ip.title),
                "read special", io::message::notice);
      }
    }    
  } // DIHREST

  // J-Value restraints
  if (sim.param().jvalue.mode != simulation::jvalue_restr_off){
    io::igzstream jval_file;
    
    if (args.count("jval") != 1){
      io::messages.add("jvalue restraints: no data file specified (use @jval)",
		       "read special", io::message::error);
    } else {
      jval_file.open(args["jval"].c_str());
      if (!jval_file.is_open()){
	io::messages.add("opening jvalue restraints file failed!\n",
			 "read_special", io::message::error);
      } else {
        io::In_Jvalue ij(jval_file);
        ij.quiet = quiet;

        ij.read(topo, conf, sim, os);
        io::messages.add("jvalue restraints read from " + args["jval"] + "\n" +
                util::frame_text(ij.title),
                "read special", io::message::notice);
      }
    }
  } // JVALUE

    // Xray restraints
  if (sim.param().xrayrest.xrayrest != simulation::xrayrest_off){
    io::igzstream xray_file;

    if (args.count("xray") != 1){
      io::messages.add("xray restraints: no data file specified (use @xray)",
		       "read special", io::message::error);
    } else {
      xray_file.open(args["xray"].c_str());
      if (!xray_file.is_open()){
	io::messages.add("opening xray restraints file failed!\n",
			 "read_special", io::message::error);
      } else {
        io::In_Xrayresspec ix(xray_file);
        ix.quiet = quiet;

        ix.read(topo, sim, os);
        io::messages.add("xray restraints read from " + args["xray"] + "\n" +
                util::frame_text(ix.title),
                "read special", io::message::notice);
      }
    }
  } // XRAY

    // FRICTION
  if (sim.param().stochastic.sd && sim.param().stochastic.ntfr == 2){
    io::igzstream friction_file;
  
    if (args.count("friction") != 1){
      io::messages.add("friction specification: no data file specified (use @friction)",
		       "read special", io::message::error);
    }
    else{
      friction_file.open(args["friction"].c_str());
      if (!friction_file.is_open()) {
        io::messages.add("opening friction file failed!\n",
                "read_special", io::message::error);
      } else {
        io::In_Friction infr(friction_file);
        infr.quiet = quiet;

        infr.read(topo, sim, os);
        io::messages.add("atomic friction coefficients read from " + args["friction"] +
                "\n" + util::frame_text(infr.title),
                "read special", io::message::notice);
      }
    }
  } // FRICTION

  // LEUS
  if (sim.param().localelev.localelev != simulation::localelev_off) {
    io::igzstream led_file;
    if (args.count("led") != 1) {
      io::messages.add("LEUS: no definition file specified (use @led)",
              "read special", io::message::error);
    } else {
      led_file.open(args["led"].c_str());
      if (!led_file.is_open()) {
        io::messages.add("opening of LEUS definition file failed!\n",
                "read_special", io::message::error);
      } else {
        io::In_Localelevspec inled(led_file);
        inled.quiet = quiet;

        inled.read(topo, sim, os);
        io::messages.add("LEUS coordinate definition read from " + args["led"] + "\n" +
                util::frame_text(inled.title), "read special", io::message::notice);
      }
    } // LED

    if (sim.param().localelev.read) {
      io::igzstream lud_file;
      if (args.count("lud") != 1) {
        io::messages.add("LEUS: no database file specified (use @lud)",
                "read special", io::message::error);
      } else {
        lud_file.open(args["lud"].c_str());
        if (!lud_file.is_open()) {
          io::messages.add("opening of LEUS database file failed!\n",
                  "read_special", io::message::error);
        } else {
          io::In_LEUSBias inlud(lud_file);
          inlud.quiet = quiet;

          inlud.read(topo, conf, sim, os);
          io::messages.add("LEUS umbrella database file read from " + args["lud"] +
                  "\n" + util::frame_text(inlud.title),
                  "read special", io::message::notice);
        }
      } // LUD
    } // if external LUD
  } // LEUS

  // QMMM
  if (sim.param().qmmm.qmmm != simulation::qmmm_off) {
    io::igzstream qmmm_file;

    if (args.count("qmmm") != 1) {
      io::messages.add("QM/MM: no specification file specified (use @qmmm)",
              "read special", io::message::error);
    } else {
      qmmm_file.open(args["qmmm"].c_str());
      if (!qmmm_file.is_open()) {
        io::messages.add("opening QM/MM specification file failed!\n",
                "read_special", io::message::error);
      } else {
        io::In_QMMM iq(qmmm_file);
        iq.quiet = quiet;

        iq.read(topo, sim, os);
        io::messages.add("QM/MM specifciation read from " +
                args["qmmm"] + "\n" + util::frame_text(iq.title),
                "read special", io::message::notice);
      }
    }
  }

  // check for errors and abort
  if (io::messages.contains(io::message::error) ||
      io::messages.contains(io::message::critical))
    return -1;
  
  return 0;
}

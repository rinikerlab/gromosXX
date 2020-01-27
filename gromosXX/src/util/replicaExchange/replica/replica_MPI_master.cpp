/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   replica_MPI_master.cpp
 * Author: bschroed
 * 
 * Created on November 8, 2019, 12:48 PM
 */

#include "replica_MPI_master.h"
#include <io/argument.h>
#include <util/error.h>
#include <util/debug.h>
#include <math/volume.h>

#ifdef XXMPI
    #include <mpi.h>
#endif

#undef MODULE
#undef SUBMODULE
#define MODULE util
#define SUBMODULE replica_exchange

util::replica_MPI_Master::replica_MPI_Master(io::Argument _args, int cont,  int globalThreadID, 
        int simulation_rank, int simulation_ID, int simulation_num_threads, simulation::mpi_control_struct replica_mpi_control) : replica_Interface( globalThreadID, replica_mpi_control, _args){
 #ifdef XXMPI

    /**
     * Build up replica - reads in the input again and build output files.
     * 
     * @param _args
     * @param cont
     * @param _ID
     * @param _rank
     * @param simulation_rank
     * @param simulation_ID
     * @param simulation_num_threads
     */
    
    MPI_DEBUG(4, "replica_MPI_MASTER "<< globalThreadID <<":Constructor:\t  "<< globalThreadID <<":\t START");
    
    /**
     * READ INPUT
     */
    // do continuation run?
    // change name of input coordinates
    if(cont == 1){
      std::multimap< std::string, std::string >::iterator it = args.lower_bound(("conf"));
      size_t pos = (*it).second.find_last_of(".");
      std::stringstream tmp;
      tmp << "_" << (simulation_ID+1);
      (*it).second.insert(pos, tmp.str());
    }
    
    MPI_DEBUG(5, "replica_MPI_MASTER "<< rank <<":Constructor:\t  "<< globalThreadID <<":\t start read in");
    //Build structure
    sim.mpi = true;
    sim.mpi_control = replica_mpi_control;  //build MPI parallelism
    
    if (io::read_input(args, topo, conf, sim, md, *os, true)) { 
      io::messages.display(*os);
      std::cerr << "\nErrors during initialization!\n" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, E_INPUT_ERROR);
    }
    MPI_DEBUG(5, "replica_MPI_MASTER "<< globalThreadID <<":Constructor:\t  "<< globalThreadID <<":\t REad in input already");
    
    MPI_DEBUG(5, "replica_MPI_MASTER "<< globalThreadID << " now make me master ");
    /**
     * INIT OUTPUT
     */
    // set output file
    std::stringstream tmp;
    tmp << "_" << (simulationID+1);
    std::string out;
    std::multimap< std::string, std::string >::iterator it = args.lower_bound(("repout"));
    size_t pos = (*it).second.find_last_of(".");
    (*it).second.insert(pos, tmp.str());
    os = new std::ofstream((*it).second.c_str());

    util::print_title(true, *os, true); // printing read in.

    // set trajectory
    std::stringstream trajstr;
    trajstr << GROMOSXX << "\n\tReplica Exchange with Replica ID " << (simulation_ID+1) << std::endl;
    std::string trajname = trajstr.str();

    traj = new io::Out_Configuration(trajname, *os);
    
    // manipulate trajectory files
    // just inserting ID: NAME_ID.cnf
    std::string fin;
    it = args.lower_bound(("fin"));
    pos = (*it).second.find_last_of(".");
    (*it).second.insert(pos, tmp.str());

    if (sim.param().write.position && args.count("trc") > 0) {
      it = args.lower_bound("trc");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if (sim.param().write.energy && args.count("tre") > 0) {
      it = args.lower_bound("tre");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if (sim.param().write.free_energy && args.count("trg") > 0) {
      it = args.lower_bound("trg");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if (sim.param().write.velocity && args.count("trv") > 0) {
      it = args.lower_bound("trv");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if ((sim.param().polarise.write || sim.param().jvalue.write || sim.param().xrayrest.write
            || sim.param().distanceres.write || sim.param().distancefield.write
            || sim.param().dihrest.write || sim.param().localelev.write
            || sim.param().electric.dip_write || sim.param().electric.cur_write
            || sim.param().addecouple.write || sim.param().nemd.write
            || sim.param().orderparamrest.write || sim.param().print.monitor_dihedrals ) && args.count("trs") > 0) {
      it = args.lower_bound("trs");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if (sim.param().write.force && args.count("trf") > 0) {
      it = args.lower_bound("trf");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if (sim.param().write.block_average && sim.param().write.energy && args.count("bae") > 0) {
      it = args.lower_bound("bae");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    if (sim.param().write.block_average && sim.param().write.free_energy && args.count("bag") > 0) {
      it = args.lower_bound("bag");
      pos = (*it).second.find_last_of(".");
      (*it).second.insert(pos, tmp.str());
    }
    
    
    // Chris: setting the title after init does not make much sense. The init function already prints it
    std::stringstream trajtitle;
    trajtitle << GROMOSXX << "\n" << sim.param().title << "\n\tReplica " << (simulation_ID+1) << "on Node " << rank;
    
    traj->title(trajtitle.str());
    traj->init(args, sim.param());

    *os << "\nMESSAGES FROM INITIALISATION\n";
    if (io::messages.display(*os) >= io::message::error) {
      *os << "\nErrors during initialization!\n" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, E_INPUT_ERROR);
    }

    // random generator
    std::stringstream seed;
    seed << sim.param().start.ig*ID;
    rng = new math::RandomGeneratorGSL(seed.str(), -1);

    *os << "==================================================\n"
        << " MAIN MD LOOP\n"
        << "==================================================\n\n";

      DEBUG(5, "replica_MPI_MASTER "<< globalThreadID <<":Constructor:\t Temp of replica  "<< globalThreadID <<": " << simulation_ID << " \t" << sim.param().multibath.multibath.bath(0).temperature);
      MPI_DEBUG(4, "replica_MPI_MASTER "<< globalThreadID <<":Constructor:\t replica Constructor  "<< globalThreadID <<": \t DONE");
#else
    throw "Can not construct Replica_MPI as MPI is not enabled!";
#endif    
}

util::replica_MPI_Master::~replica_MPI_Master() {
  delete rng;
  delete traj;
  delete os;
}

 #ifdef XXMPI
void util::replica_MPI_Master::run_MD(){
    // run MD simulation
    int error;
    
    //next_stepf for mpi slaves  
    int next_step = 1;  //bool that signalises if next step is fine.

    while ((unsigned int)(sim.steps()) <  stepsPerRun + curentStepNumber) {
      DEBUG(5, "replica_MPI "<< globalThreadID <<":run_MD:\t Start");      
      traj->write(conf, topo, sim, io::reduced);
      // run a step
      DEBUG(5, "replica_MPI "<< globalThreadID <<":run_MD:\t simulation!:");
        if ((error = md.run(topo, conf, sim))) {
            switch (error) {
              case E_SHAKE_FAILURE:
                std::cerr << "SHAKE FAILURE in Replica " << (simulationID+1) << " on node " << globalThreadID << std::endl;
                io::messages.display();
                MPI_Abort(MPI_COMM_WORLD, error);
                break;
              case E_SHAKE_FAILURE_SOLUTE:
                std::cerr << "SHAKE FAILURE SOLUTE in Replica " << (simulationID+1) << " on node " << globalThreadID << std::endl;
                io::messages.display();
                MPI_Abort(MPI_COMM_WORLD, error);
                break;
              case E_SHAKE_FAILURE_SOLVENT:
                std::cerr << "SHAKE FAILURE SOLVENT in Replica " << (simulationID+1) << " on node " << globalThreadID << std::endl;
                io::messages.display();
                MPI_Abort(MPI_COMM_WORLD, error);
                break;
              case E_NAN:
                std::cerr << "NAN error in Replica " << (simulationID+1) << " on node " << globalThreadID << std::endl;
                io::messages.display();
                MPI_Abort(MPI_COMM_WORLD, error);
                break;
              default:
                std::cerr << "Unknown error in Replica " << (simulationID+1) << " on node " << globalThreadID << std::endl;
                io::messages.display();
                MPI_Abort(MPI_COMM_WORLD, error);
                break;
            }
            error = 0; // clear error condition
            break;
        }
      
        // tell the slaves to continue
        MPI_Bcast(&next_step, 1, MPI::INT, sim.mpi_control.simulationMasterThreadID, sim.mpi_control.simulationCOMM);

        DEBUG(5, "replica "<< globalThreadID <<":run_MD:\t clean up:");      
        traj->print(topo, conf, sim);

        ++sim.steps();
        sim.time() = sim.param().step.t0 + sim.steps() * sim.time_step_size();
    } // main md loop
    
    DEBUG(4, "replica "<< globalThreadID <<":run_MD:\t  DONE:");      

    // print final data of run
    if (curentStepNumber ==  totalStepNumber) {
      traj->print_final(topo, conf, sim);
    }
}
#endif    

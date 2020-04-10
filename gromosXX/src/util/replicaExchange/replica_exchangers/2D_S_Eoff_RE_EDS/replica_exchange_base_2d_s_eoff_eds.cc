/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   replica_exchange_base_2d_s_eoff_eds.cc
 * Author: theosm
 *
 * Created on March 29, 2020, 11:08 AM
 */
#include "util/replicaExchange/replica_mpi_tools.h"
#include <util/replicaExchange/replica_exchangers/2D_S_Eoff_RE_EDS/replica_exchange_base_2d_s_eoff_eds.h>

//Constructor
#include <util/replicaExchange/replica_exchangers/replica_exchange_base_interface.h>
#include <util/replicaExchange/replica/replica.h>
#include "replicaExchange/replica/replica_MPI_master.h"
#include "replicaExchange/replica/replica_MPI_slave.h"
#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <algorithm/algorithm/algorithm_sequence.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <io/argument.h>
#include <util/usage.h>
#include <util/error.h>

#include <io/read_input.h>
#include <io/print_block.h>

#include <time.h>
#include <unistd.h>

#include <io/configuration/out_configuration.h>
#include <math/random.h>
#include <math/volume.h>
#include <string>

#ifdef XXMPI
#include <mpi.h>
#endif

#undef MODULE
#undef SUBMODULE
#define MODULE util
#define SUBMODULE replica_exchange

/*
 * ADDITIONAL FUNCTIONS for REDS
 */

util::replica_exchange_base_2d_s_eoff_eds::replica_exchange_base_2d_s_eoff_eds(io::Argument _args,
                                                            unsigned int cont,
                                                            unsigned int globalThreadID,
                                                            replica_graph_mpi_control replicaGraphMPIControl,
                                                            simulation::mpi_control_struct replica_mpi_control):
                            replica_exchange_base_interface(_args, cont, globalThreadID, replicaGraphMPIControl, replica_mpi_control),
                            reedsParam(replica->sim.param().reeds)
{
    MPI_DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":Constructor:\t START" );
    DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":Constructor:\t simID "<<simulationID);

    //RE-Vars
    MPI_DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":Constructor:\t setParams" );
    setParams();

    MPI_DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":Constructor:\t DONE");
}

util::replica_exchange_base_2d_s_eoff_eds::~replica_exchange_base_2d_s_eoff_eds() {
    delete replica;
}

void util::replica_exchange_base_2d_s_eoff_eds::setParams(){
    // set some variables
    stepsPerRun = replica->sim.param().step.number_of_steps;
    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t NUMBER OF STEPS "<<stepsPerRun);

    run = 0;
    total_runs = replica->sim.param().replica.trials + replica->sim.param().replica.equilibrate;
    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t NUMBER OF total_runs "<<total_runs);

    partnerReplicaID = simulationID;
    time = replica->sim.time();
    steps = 0;
    switched = 0;
    replica->curentStepNumber=0;
    replica->totalStepNumber = total_runs*stepsPerRun;
    replica->stepsPerRun= stepsPerRun;

    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t PARAM START");

    T = replica->sim.param().reeds.temperature;
    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t got  T " << T);
    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t got simulationID: "<< simulationID);

    //set position info
    replica->sim.param().reeds.eds_para[simulationID].pos_info = std::make_pair(simulationID, simulationID);
    DEBUG(1, "BASE Constructor with simulationID, replica->pos_info= " << simulationID << ", "
    << replica->sim.param().reeds.eds_para[simulationID].pos_info.first << ", "
    << replica->sim.param().reeds.eds_para[simulationID].pos_info.second << "\n");

    pos_info = replica->sim.param().reeds.eds_para[simulationID].pos_info;
    DEBUG(1, "BASE Constructor with simulationID, pos_info= " << simulationID << ", "
    << pos_info.first << ", " << pos_info.second << "\n");

    //just to check -- theosm
    std::pair<int, int> a = reedsParam.eds_para[simulationID].pos_info;
    DEBUG(1, "JUST TO CHECK: BASE Constructor with simulationID, reedsParam->pos_info= " << simulationID << ", "
    << a.first << ", " << a.second << "\n");

    set_s();
    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t got s" << l);

    dt = replica->sim.param().step.dt;
    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t dt " <<dt);

    MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":setParams:\t PARAM DONE ");
}

void util::replica_exchange_base_2d_s_eoff_eds::set_s() {
  MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":set_s:\t START ");

  eds_para = replica->sim.param().reeds.eds_para[simulationID];
  replica->sim.param().eds = eds_para;
  MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":set_s:\t eds_para s size: " << replica->sim.param().eds.s.size());

  l = replica->sim.param().eds.s[0];    //todoAssume only 1s EDS
  MPI_DEBUG(4,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":set_s:\t DONE " );
}

void util::replica_exchange_base_2d_s_eoff_eds::init() {
  DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: INIT\n\n");
  MPI_DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":init:\t init \t START");
  DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":init:\t start init from baseclass \t NEXT");
  //replica->init();
  DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":init:\t init_eds_stat \t NEXT");
  init_eds_stat();
  MPI_DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":init:\t DONE");
}

//initialize output files
void util::replica_exchange_base_2d_s_eoff_eds::init_eds_stat(){
        DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: INIT_EDS_STAT\n\n");
        DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":init_eds_stat:\t START");

        ID_t currentID=1000; //error value
        currentID = simulationID;
        replicaStatData[currentID].ID=currentID;
        //assignment of position info of each replica
        replicaStatData[currentID].pos_info.first = pos_info.first;
        replicaStatData[currentID].pos_info.second = pos_info.second;
        DEBUG(3, "init_eds_stat(), replicaStatData[currentID].pos_info.first= " << replicaStatData[currentID].pos_info.first
        << " with currentID= " << replicaStatData[currentID].ID << "\n");
        DEBUG(3, "init_eds_stat(), replicaStatData[currentID].pos_info.second= " << replicaStatData[currentID].pos_info.second
        << " with currentID= " << replicaStatData[currentID].ID << "\n");
        replicaStatData[currentID].T=T;
        replicaStatData[currentID].s=l; //l==s because of the implementation of hamiltonian replica exchange.
        replicaStatData[currentID].dt=dt;
        replicaStatData[currentID].run=0;
        replicaStatData[currentID].epot_vec.resize(replicaGraphMPIControl.numberOfReplicas);
        replicaStatData[currentID].prob_vec.resize(replicaGraphMPIControl.numberOfReplicas);

        DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":init_eds_stat:\t DONE");
}

//RE
void util::replica_exchange_base_2d_s_eoff_eds::swap(){
    DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":swap:\t START");

    //for(int trial=0; trial < 4; ++trial){
      //idea: s-dim: 1st & 3rd trial changing
      //2nd trial eoff-dim

      partnerReplicaID = find_partner();
      DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: SWAP\n\n");

      //exchanging coord_ID's
      DEBUG(3, "swap(): simulationID, partnerReplicaID= " << simulationID << ", " << partnerReplicaID << "\n");
      //replica->sim.param().reeds.eds_para[simulationID].pos_info.second = partnerReplicaID;
      replica->sim.param().reeds.eds_para[partnerReplicaID].pos_info.second = simulationID;

      //theosm: tried several ways -- might be an alternative
      /*
      int tmp = replica->sim.param().reeds.eds_para[partnerReplicaID].pos_info.second;
      replica->sim.param().reeds.eds_para[partnerReplicaID].pos_info.second = replica->sim.param().reeds.eds_para[simulationID].pos_info.second;
      replica->sim.param().reeds.eds_para[simulationID].pos_info.second = tmp;

      DEBUG(1, "swap(): simulationID, reedsParam->pos_info.second= " << simulationID << ", "
      << replica->sim.param().reeds.eds_para[simulationID].pos_info.second << "\n");
      DEBUG(1, "swap(): partnerReplicaID, reedsParam->pos_info.second= " << partnerReplicaID << ", "
      << replica->sim.param().reeds.eds_para[partnerReplicaID].pos_info.second << "\n");
      */

      if (partnerReplicaID != simulationID) // different replica?
      {
        if(run % 2 == 1){
          swap_s(partnerReplicaID);
        }
        else{
        swap_eoff(partnerReplicaID);
        }
        if (switched) {
          if (globalThreadID < partnerReplicaID) {
            send_coord(partnerReplicaID);
            receive_new_coord(partnerReplicaID);
            // the averages of current and old are interchanged after calling exchange_state() and have to be switched back
            exchange_averages();
          } else {
            receive_new_coord(partnerReplicaID);
            send_coord(partnerReplicaID);
            // the averages of current and old are interchanged after calling exchange_state() and have to be switched back
            exchange_averages();
          }
        }
      }
      else {  // no exchange with replica itself
        probability = 0.0;
        switched = 0;
      }
      if(switched && replica->sim.param().replica.scale) {
        velscale(partnerReplicaID);
      }
    //}

  DEBUG(3,"replica_exchange_base_2d_s_eoff_eds "<< globalThreadID <<":swap:\t DONE");
}

void util::replica_exchange_base_2d_s_eoff_eds::swap_s(const unsigned int partnerReplicaID) {
  DEBUG(4, "replica "<<  globalThreadID <<":swap:\t  START");

  DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: SWAP_S\n\n");

  unsigned int partnerReplicaMasterThreadID = partnerReplicaID;
  unsigned int numReps = replica->sim.param().reeds.num_l * replica->sim.param().reeds.num_eoff;

  // does partner exist?
  if (partnerReplicaID < numReps && partnerReplicaID != simulationID) {
    // the one with lower ID does probability calculation
    if (simulationID < partnerReplicaID) {

      // posts a MPI_Recv(...) matching the MPI_Send below
      probability = calc_probability(partnerReplicaID);
      const double randNum = rng.get();

      std::vector<double> prob(2);
      prob[0] = probability;
      prob[1] = randNum;

#ifdef XXMPI
      MPI_Send(&prob[0], 2, MPI_DOUBLE, partnerReplicaMasterThreadID, SENDCOORDS, replicaGraphMPIControl.comm);
#endif

      if (randNum < probability) {
        switched = true;
      } else
        switched = false;
    } else {    //The Partner sends his data to The calculating Thread
      //special case if lambda also needs to be exchanged
      bool sameLambda = (l == replica->sim.param().replica.lambda[partnerReplicaID]);
      DEBUG(1, "swap_s: simID, s value= " << simulationID << ", " << l << "\n");
      DEBUG(1,"swap_s: simID, bool sameLambda= " << simulationID << ", " << sameLambda << "\n");
      if(!sameLambda){      //exchange LAMBDA
        // E21: Energy with configuration 2 and lambda 1(of partner)
        const double E21 = calculate_energy(partnerReplicaMasterThreadID);
        // this we can store as the partner energy of the current replica
        epot_partner = E21;
        // E22: Energy with configuration 2 and lambda 2(own one)
#ifdef XXMPI
        const double E22 = epot;
        // send E21 and E22
        double energies[2] = {E22, E21};
        //this send operation is matched in calc_probability()
        MPI_Send(&energies[0], 2, MPI_DOUBLE, partnerReplicaMasterThreadID, SWITCHENERGIES,  replicaGraphMPIControl.comm);
#endif
      } else { // sameLambda
#ifdef XXMPI
        double energies[2] = {epot, 0.0};
        MPI_Send(&energies[0],2,MPI_DOUBLE, partnerReplicaMasterThreadID, SWITCHENERGIES,  replicaGraphMPIControl.comm);
#endif
     }
      if (replica->sim.param().pcouple.scale != math::pcouple_off) {
#ifdef XXMPI
        math::Box box_replica = replica->conf.current().box;    //exchange box
        MPI_Send(&box_replica(0)[0], 1, MPI_BOX, partnerReplicaMasterThreadID, BOX,  replicaGraphMPIControl.comm);
#endif
      }

#ifdef XXMPI
      MPI_Status status;
#endif
      std::vector<double> prob;
      prob.resize(2);
#ifdef XXMPI
      MPI_Recv(&prob[0], 2, MPI_DOUBLE, partnerReplicaMasterThreadID, SENDCOORDS,  replicaGraphMPIControl.comm, &status);
#endif
      //Have we been exchanged little partner?
      probability = prob[0];
      double randNum = prob[1];

      if (randNum < probability) {
        switched = true;
      } else {
        switched = false;
      }
    }

  } else {//This should be an error!
      throw "Partner does not exist!";
    /*
      partner = ID;
    switched = false;
    probability = 0.0;

    */
  }
    DEBUG(4, "replica "<< globalThreadID <<":swap:\t  DONE");
}

void util::replica_exchange_base_2d_s_eoff_eds::swap_eoff(const unsigned int partnerReplicaID) {
  DEBUG(4, "replica "<<  globalThreadID <<":swap:\t  START");

  DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: SWAP_EOFF\n\n");

  unsigned int partnerReplicaMasterThreadID = partnerReplicaID;
  unsigned int numReps = replica->sim.param().reeds.num_l * replica->sim.param().reeds.num_eoff;

  // does partner exist?
  if (partnerReplicaID < numReps && partnerReplicaID != simulationID) {
    // the one with lower ID does probability calculation
    if (simulationID < partnerReplicaID) {

      // posts a MPI_Recv(...) matching the MPI_Send below
      probability = calc_probability(partnerReplicaID);
      const double randNum = rng.get();

      std::vector<double> prob(2);
      prob[0] = probability;
      prob[1] = randNum;

#ifdef XXMPI
      MPI_Send(&prob[0], 2, MPI_DOUBLE, partnerReplicaMasterThreadID, SENDCOORDS, replicaGraphMPIControl.comm);
#endif

      if (randNum < probability) {
        switched = true;
      } else
        switched = false;
    } else {    //The Partner sends his data to The calculating Thread
      //special case if lambda also needs to be exchanged
      bool sameLambda = (l == replica->sim.param().replica.lambda[partnerReplicaID]);
      DEBUG(1, "swap_eoff: simID, s value= " << simulationID << ", " << l << "\n");
      DEBUG(1,"swap_eoff: simID, bool sameLambda= " << simulationID << ", " << sameLambda << "\n");
      if(!sameLambda){      //exchange LAMBDA
        // E21: Energy with configuration 2 and lambda 1(of partner)
        const double E21 = calculate_energy(partnerReplicaMasterThreadID);
        // this we can store as the partner energy of the current replica
        epot_partner = E21;
        // E22: Energy with configuration 2 and lambda 2(own one)
#ifdef XXMPI
        const double E22 = epot;
        // send E21 and E22
        double energies[2] = {E22, E21};
        //this send operation is matched in calc_probability()
        MPI_Send(&energies[0], 2, MPI_DOUBLE, partnerReplicaMasterThreadID, SWITCHENERGIES,  replicaGraphMPIControl.comm);
#endif
      } else { // sameLambda
#ifdef XXMPI
        double energies[2] = {epot, 0.0};
        MPI_Send(&energies[0],2,MPI_DOUBLE, partnerReplicaMasterThreadID, SWITCHENERGIES,  replicaGraphMPIControl.comm);
#endif
     }
      if (replica->sim.param().pcouple.scale != math::pcouple_off) {
#ifdef XXMPI
        math::Box box_replica = replica->conf.current().box;    //exchange box
        MPI_Send(&box_replica(0)[0], 1, MPI_BOX, partnerReplicaMasterThreadID, BOX,  replicaGraphMPIControl.comm);
#endif
      }

#ifdef XXMPI
      MPI_Status status;
#endif
      std::vector<double> prob;
      prob.resize(2);
#ifdef XXMPI
      MPI_Recv(&prob[0], 2, MPI_DOUBLE, partnerReplicaMasterThreadID, SENDCOORDS,  replicaGraphMPIControl.comm, &status);
#endif
      //Have we been exchanged little partner?
      probability = prob[0];
      double randNum = prob[1];

      if (randNum < probability) {
        switched = true;
      } else {
        switched = false;
      }
    }

  } else {//This should be an error!
      throw "Partner does not exist!";
    /*
      partner = ID;
    switched = false;
    probability = 0.0;

    */
  }
    DEBUG(4, "replica "<< globalThreadID <<":swap:\t  DONE");
}

int util::replica_exchange_base_2d_s_eoff_eds::find_partner() const {
  unsigned int num_eoff = replica->sim.param().reeds.num_eoff;
  DEBUG(3,"find_partner: num_eoff= " << num_eoff << "\n");
  unsigned int num_l = replica->sim.param().reeds.num_l;
  DEBUG(3,"find_partner: num_l= " << num_l << "\n");
  unsigned int numT = replica->sim.param().replica.num_T;
  DEBUG(3,"find_partner: numT= " << numT << "\n");

  unsigned int numReps = num_l * num_eoff;
  DEBUG(3,"find_partner: numReps= " << numReps << "\n");


  unsigned int ID = simulationID;

  DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: FIND_PARTNER\n\n");

  unsigned int partner = ID;
  bool even = ID % 2 == 0;
  bool evenRow = (ID % num_l) % 2 == 0;//1st row is here the 0th row and therefore even!
  bool evenCol = (ID / num_l) % 2 == 0;//1st col is here the 0th col and therefore even!
  bool numEoffeven = num_eoff % 2 == 0;
  bool periodic = replica->sim.param().reeds.periodic;

  //theosm
  //edge cases for s dimension
  bool upper = ID % num_l == 0;
  bool lower = ID % num_l == num_l - 1;
  //current s coord == j € [0, num_l -1)
  unsigned int j = ID % num_l;
  //edge cases for eoff dimension
  bool left_edge = ID == j;
  bool right_edge = ID == (numReps - num_l + j);
  DEBUG(3,"ID, j, upper, lower, left_edge, right_edge= " << ID << ", " << j << ", " << upper << ", " << lower
  << ", " << left_edge << ", " << right_edge << "\n");

  //theosm: on my own
  /*
  switch ((run % 4) - 1) {
    case 0: //s dimension
    DEBUG(1,"find_partner: FIRST case\n");
        if (even){
          partner = ID + 1;
          //edge case
          if(lower)
            partner = ID;
        }
        else{
          partner = ID - 1;
          //edge case
          if(upper)
            partner = ID;
        }
    DEBUG(1,"find_partner(first case): partner of ID=" << ID << " is " << partner << "\n");
      break;

    case 1: //eoff dimension
    DEBUG(1,"find_partner: SECOND case -- nothing to be done right now\n");
      partner = ID + num_l;
      //edge case
      if(right_edge)
        partner = ID;

      partner = ID - num_l;
      //edge case
      if(left_edge)
        partner = ID;
    DEBUG(1,"find_partner(second case): partner of ID=" << ID << " is " << partner << "\n");
      break;

    case 2: //s dimension
    DEBUG(1,"find_partner: THIRD case\n");
      //if (numEoffeven) {
        if (even){
          partner = ID + 1;
          //edge case
          if(lower)
            partner = ID;
        }
        else{
          partner = ID - 1;
          //edge case
          if(upper)
          partner = ID;
        }
    DEBUG(1,"find_partner(third case): partner of ID=" << ID << " is " << partner << "\n");
      break;

    case -1: //eoff dimension
    DEBUG(1,"\n find_partner: FOURTH case -- nothing to be done right now\n");
      partner = ID - num_l;
      //edge case
      if(left_edge)
        partner = ID;

      partner = ID + num_l;
      //edge case
      if(right_edge)
        partner = ID;
    DEBUG(1,"\n find_partner(fourth case): partner of ID=" << ID << " is " << partner << "\n");
      break;
  }
  */



    // determine switch direction -- already given just modified
    switch ((run % 4) - 1) {
      case 0: //s dimension
      DEBUG(5,"find_partner: FIRST case\n");
        if (numEoffeven) {
          if (even) {
            partner = ID + 1;
            DEBUG(1,"\nHERE0A\n");
            DEBUG(1,"\nHERE2A\n");
            //edge case
            if(lower)
              partner = ID;
          }
          else {
            partner = ID - 1;
            DEBUG(1,"\nHERE1A\n");
            DEBUG(1,"\nHERE3A\n");
            //edge case
            if(upper)
              partner = ID;
          }
        } else {
          if (evenRow) {
            if (even) {
              partner = ID + 1;
              DEBUG(1,"\nHERE0\n");
              DEBUG(1,"\nHERE2\n");
              DEBUG(1,"\nHERE6\n");
              DEBUG(1,"\nHERE8\n");
              //edge case
              if(lower)
                partner = ID;
            }
            else {
              partner = ID + 1;
              DEBUG(1,"\nHERE3\n");
              DEBUG(1,"\nHERE5\n");
              //edge case
              if(lower)
                partner = ID;
            }
          } else {
            if (even) {
              partner = ID - 1;
              DEBUG(1,"\nHERE4\n");
              //edge case
              if(upper)
                partner = ID;
            }
            else {
              partner = ID - 1;
              DEBUG(1,"\nHERE1\n");
              DEBUG(1,"\nHERE7\n");
              //edge case
              if(upper)
                partner = ID;
            }
          }
        }
      DEBUG(1,"find_partner(first case): partner of ID=" << ID << " is " << partner << "\n");
        break;

      case 1: //eoff dimension
      DEBUG(5,"find_partner: SECOND case\n");
        if (evenCol) {
          partner = ID + num_l;
          DEBUG(1,"\nHERE4A\n");
          DEBUG(1,"\nHERE6A\n");
          //edge case
          if(right_edge && !periodic) partner = ID;
          if(right_edge && periodic) {partner = (ID + num_l) % numReps; DEBUG(1,"\nPERIODIC\n");}
          }
        else {
          partner = ID - num_l;
          DEBUG(1,"\nHERE5A\n");
          DEBUG(1, "\nHERE7A\n");
          //edge case
          if(left_edge && !periodic) partner = ID;
          if(left_edge && periodic) {partner = ID + (numReps - num_l); DEBUG(1,"\nPERIODIC\n");}
        }
      DEBUG(1,"find_partner(second case): partner of ID=" << ID << " is " << partner << "\n");
        break;

      case 2: //s dimension
      DEBUG(5,"find_partner: THIRD case\n");
        if (numEoffeven) {
          if (even) {
            partner = ID + 1;
            //edge case
            if(lower)
              partner = ID;
          }
          else {
            partner = ID - 1;
            //edge case
            if(upper)
              partner = ID;
          }
        } else {
          if (evenRow) {
            if (even) {
              partner = ID + 1;
              //edge case
              if(lower)
                partner = ID;
            }
            else {
              partner = ID - 1;
              //edge case
              if(upper)
                partner = ID;
            }
          } else {
            if (even) {
              partner = ID + 1;
              //edge case
              if(lower)
                partner = ID;
            }
            else {
              partner = ID - 1;
              //edge case
              if(upper)
                partner = ID;
            }
          }
        }
      DEBUG(1,"find_partner(third case): partner of ID=" << ID << " is " << partner << "\n");
        break;

      case -1: //eoff dimension
      DEBUG(5,"find_partner: FOURTH case\n");
        if (evenCol) {
          partner = ID - num_l;
          //edge case
          if(left_edge && !periodic) partner = ID;
          if(left_edge && periodic) {partner = ID + (numReps - num_l); DEBUG(1,"\nPERIODIC\n");}
        }
        else {
          partner = ID + num_l;
          //edge case
          if(right_edge && !periodic) partner = ID;
          if(right_edge && periodic) {partner = (ID + num_l) % numReps; DEBUG(1,"\nPERIODIC\n");}
        }
      DEBUG(1,"find_partner(fourth case): partner of ID=" << ID << " is " << partner << "\n");
        break;
    }

  /*
  // partner out of range ? - Do we really need this or is it more a hack hiding bugs?
  if (partner > numT * num_l - 1)
    partner = ID;
  */

  return partner;
}

////exchange params
void util::replica_exchange_base_2d_s_eoff_eds::reset_eds() {//only reset switched parameters of change_eds() function
  DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: RESET_EDS\n\n");
  replica->sim.param().eds = eds_para;
  replica->sim.param().step.dt = dt;
  replica->conf.current().force= force_orig;
  replica->conf.current().virial_tensor= virial_tensor_orig;
}

void util::replica_exchange_base_2d_s_eoff_eds::change_eds(const unsigned int partner){//only change parameters, which are needed for energy calculation i.e.

  DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: CHANGE_EDS\n\n");
  int idx;
  if (replica->sim.param().reeds.num_l == 1){
    idx = 0;
  }
  else{
    idx = partner;
  }

  replica->sim.param().step.dt = replica->sim.param().reeds.dt[idx];
  replica->sim.param().eds= replica->sim.param().reeds.eds_para[idx];
  force_orig = replica->conf.current().force;
  virial_tensor_orig = replica->conf.current().virial_tensor;
}

////calc exchange Energy
 /*
 * calc_energy_eds_stat() is only used for statistical purposes in eds_stat()
 * In order to avoid any adjustment of the mpi communication and thus reducing the complexity, the
 * energy_calculation and probability calculations from replica.cc are not adjusted to work
 * for non-pairwise exchanges. Instead, calc_energy_eds_stat() calculates the potential energy
 * of the current configuration for a new smoothing parameter s.
 * The exchange probabilities can be calculated in a postprocessing step, using these energies
 * given in the energy_stat output files.
 */
double util::replica_exchange_base_2d_s_eoff_eds::calc_energy_eds_stat(double s){
    double old_dt;
    double old_s;
    double old_eds_vr;
    algorithm::Algorithm * ff;
    DEBUG(5,"\n\nreplica_exchange_base_2d_s_eoff_eds: CALC_ENERGY_EDS_STAT\n\n");
    if(replica->sim.param().eds.eds){
          //to reset old state
          old_dt=replica->sim.param().step.dt;
          old_s=replica->sim.param().eds.s[0];
          old_eds_vr=replica->conf.current().energies.eds_vr;
          force_orig = replica->conf.current().force;
          virial_tensor_orig = replica->conf.current().virial_tensor;
          //only temporary change
          replica->sim.param().eds.s[0]=s;

          ff = replica->md.algorithm("EDS");
    }
    else {
          print_info("eds_stat() i.e calc_energy_eds_stat() called for non EDS simulation!");
      #ifdef XXMPI
          MPI_Abort(MPI_COMM_WORLD, E_UNSPECIFIED);
      #endif
    }

    //Calculate energies
    if (ff->apply(replica->topo, replica->conf, replica->sim)) {
      print_info("Error in Forcefield energy calculation!");
     #ifdef XXMPI
      MPI_Abort(MPI_COMM_WORLD, E_UNSPECIFIED);
     #endif
      return 1;
    }

    double energy=replica->conf.current().energies.eds_vr;

    // reset old EDS state
    replica->conf.current().energies.eds_vr=old_eds_vr;
    replica->sim.param().eds.s[0] = old_s;
    replica->sim.param().step.dt = old_dt;
    replica->conf.current().force=force_orig;
    replica->conf.current().virial_tensor=virial_tensor_orig;

    return energy;
}

double util::replica_exchange_base_2d_s_eoff_eds::calculate_energy_core() {

    double energy = 0.0;
    algorithm::Algorithm * ff;

     ff = replica->md.algorithm("EDS");

     DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: CALCULATE_ENERGY_CORE\n\n");

    //Calculate energies
    DEBUG(5, "replica_reeds "<< globalThreadID <<":calculate_energy:\t calc energies");
    if (ff->apply(replica->topo, replica->conf, replica->sim)) {
      print_info("Error in Forcefield energy calculation!");
     #ifdef XXMPI
      MPI_Abort(MPI_COMM_WORLD, E_UNSPECIFIED);
    #endif
      return 1;
    }

    //return energies
    DEBUG(5, "replica_reeds "<< globalThreadID <<":calculate_energy"
            ":\t return energies");
    energy=replica->conf.current().energies.eds_vr;
    return energy;
}


double util::replica_exchange_base_2d_s_eoff_eds::calculate_energy(const unsigned int selectedReplicaID) {
    DEBUG(4, "replica_reeds "<< globalThreadID <<":calculate_energy:\t START");

    DEBUG(3,"\n\nreplica_exchange_base_2d_s_eoff_eds: CALCULATE_ENERGY\n\n");


    DEBUG(5, "replica_reeds "<< globalThreadID <<":calculate_energy:\t get Partner settings");
    if(selectedReplicaID!=simulationID){
        change_eds(selectedReplicaID);
    }

    double energy =  calculate_energy_core();

    if(selectedReplicaID!=simulationID){
        reset_eds();
    }
    DEBUG(4, "replica_reeds "<< globalThreadID <<":calculate_energy:\t DONE");
    return energy;
}

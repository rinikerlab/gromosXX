/* 
 * File:   replica_exchange_master.cc
 * Author: wissphil, sriniker
 * 
 * Created on April 29, 2011, 2:18 PM
 */

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
#include <util/replicaExchange/replica/replica.h>
#include <util/replicaExchange/replica_exchangers/replica_exchange_base.h>
#include <util/replicaExchange/replica_exchangers/replica_exchange_master.h>

#include <string>

#ifdef XXMPI
    #include <mpi.h>
#endif

#undef MODULE
#undef SUBMODULE
#define MODULE util
#define SUBMODULE replica_exchange

util::replica_exchange_master::replica_exchange_master(io::Argument & args,
        int cont,
        int rank,
        int simulationRank,
        int simulationID,
        int simulationThreads,
        int _size,
        int _numReplicas,
        std::vector<int> repIDs,
        std::map<ID_t, rank_t> & repMap)
:
replica_exchange_base(args, cont, rank, simulationRank, simulationID, simulationThreads, repIDs, repMap ),
size(_size),
numReplicas(_numReplicas),
repParams(replica->sim.param().replica),
repdatName(args["repdat"])
{
#ifdef XXMPI
  DEBUG(2,"replica_exchange_master "<< rank <<":Constructor:\t START");

  assert(rank == 0);
  assert(numReplicas > 0);
  DEBUG(2,"replica_exchange_master "<< rank <<":Constructor:\t rep_params THERE?");
  DEBUG(2,"replica_exchange_master "<< rank <<":Constructor:\t" << replica->sim.param().replica.num_l);
  DEBUG(2,"replica_exchange_master "<< rank <<":Constructor:\t" << replica->sim.param().replica.lambda[0]);

  assert(repParams.num_l > 0);
  
  DEBUG(4,"replica_exchange_master "<< rank <<":Constructor:\t Init Replicas \t Next");
  replicaData.resize(numReplicas);
  DEBUG(4,"replica_exchange_master "<< rank <<":Constructor:\t Replica_data type \t " << typeid(replicaData).name());

  //initialize data of replicas
  int ID = 0;
  for (int i = 0; i < repParams.num_l; ++i) {
    for (int j = 0; j < repParams.num_T; ++j) {
      replicaData[ID].ID = ID;
      replicaData[ID].T = repParams.temperature[j];
      DEBUG(4,"replica_exchange_master "<< rank <<":Constructor:\t Init Replicas \t "<< repParams.temperature[j]);
      replicaData[ID].l = repParams.lambda[i];
      replicaData[ID].dt = repParams.dt[i];
      ++ID;
    }
  }

  // set output file
 DEBUG(2,"replica_exchange_master "<< rank <<":Constructor:\t DONE");
#else
   throw "Cannot initialize replica_exchange_master without MPI!"; 
#endif
}


util::replica_exchange_master::~replica_exchange_master() {
   repOut.close();
}

void util::replica_exchange_master::receive_from_all_slaves() {
#ifdef XXMPI
  DEBUG(2,"replica_exchange_master "<< rank <<":receive_from_all_slaves:\t START\n");
  double start = MPI_Wtime();

  MPI_Status status;
  util::repInfo info;

  // receive all information from slaves
  for (unsigned int rep = 0; rep < numReplicas; ++rep) {
    unsigned int rank = repMap.find(rep)->second;
    if (rank != 0) {
      MPI_Recv(&info, 1, MPI_REPINFO, rank, REPINFO, MPI_COMM_WORLD, &status);
      replicaData[rep].run = info.run;
      replicaData[rep].epot = info.epot;
      replicaData[rep].epot_partner = info.epot_partner;
      replicaData[rep].probability = info.probability;
      replicaData[rep].switched = info.switched;
      replicaData[rep].partner = info.partner;
    }
    DEBUG(2,"replica_exchange_master "<< rank <<":receive_from_all_slaves:\t got all MPI reps\n");

    // write all information from master node to data structure
    int ID = replica->ID;
    replicaData[ID].run = replica->run;
    replicaData[ID].partner = replica->partner;
    replicaData[ID].epot = replica->epot;
    replicaData[ID].epot_partner = replica->epot_partner;
    replicaData[ID].probability = replica->probability;
    replicaData[ID].switched = replica->switched;

    /*
  for (repIterator it = replicas.begin(); it < replicas.end(); ++it) {
    int ID = (*it)->ID;
    replicaData[ID].run = (*it)->run;
    replicaData[ID].partner = (*it)->partner;
    replicaData[ID].epot = (*it)->epot;
    replicaData[ID].epot_partner = (*it)->epot_partner;
    replicaData[ID].probability = (*it)->probability;
    replicaData[ID].switched = (*it)->switched;
  }
    */
   DEBUG(2,"replica_exchange_master "<< rank <<":receive_from_all_slaves:\t " << "time used for receiving all messages: " << MPI_Wtime() - start << " seconds\n");
   DEBUG(2,"replica_exchange_master "<< rank <<":receive_from_all_slaves:\t DONE: \n");
#else
   throw "Cannot use replica_exchange_master without MPI!"; 
#endif
}
}

  
void util::replica_exchange_master::init_repOut_stat_file() {
  DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t START");
  repOut.open(repdatName.c_str());
  DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t  repdat file open ");

  repOut << "Number of temperatures:\t" << repParams.num_T << "\n"
         << "Number of lambda values:\t" << repParams.num_l << "\n";
  
  DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t set precision ");
  repOut.precision(4);
  repOut.setf(std::ios::fixed, std::ios::floatfield);
  
  DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t write Temperatures ");
  repOut << "T    \t";
  for (int t = 0; t < repParams.num_T; ++t){
    DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t it: "<<  t);
    DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t T: "<<  repParams.temperature[t]);
    repOut << std::setw(12) << repParams.temperature[t];
  }
  
  DEBUG(2,"replica_exchange_master "<< rank <<":init_repOut_stat_file:\t write lambdas ");
  repOut << "\nlambda    \t";
  for (int l = 0; l < repParams.num_l; ++l){
    repOut << std::setw(12) << repParams.lambda[l];
  }

  repOut << "\n\n";

  repOut << "#"
          << std::setw(6) << "ID"
          << " "
          << std::setw(6) << "partner"
          << std::setw(6) << "run"
          << " "
          << std::setw(13)  << "li"
          << std::setw(13)  << "Ti"
          << std::setw(18)  << "Epoti"
          << std::setw(13)  << "lj"
          << std::setw(13)  << "Tj"
          << std::setw(18)  << "Epotj"
          << std::setw(13)  << "p"
          << std::setw(6) << "exch";
  repOut << "\n";
}


void util::replica_exchange_master::write() {
   DEBUG(2,"replica_exchange_master "<< rank <<":write:\t START");

  for (unsigned int r = 0; r < numReplicas; ++r) {
    repOut << std::setw(6) << (replicaData[r].ID + 1)
            << " "
            << std::setw(6) << (replicaData[r].partner + 1)
            << std::setw(6) << replicaData[r].run
            << std::setw(13) << replicaData[r].l
            << std::setw(13) << replicaData[r].T
            << " "
            << std::setw(18) << replicaData[r].epot
            << std::setw(13) << replicaData[replicaData[r].partner].l
            << std::setw(13) << replicaData[replicaData[r].partner].T
            << " ";
    if(replicaData[r].l == replicaData[replicaData[r].partner].l)
	repOut << std::setw(18) << replicaData[replicaData[r].partner].epot;
    else
        repOut << std::setw(18) << replicaData[r].epot_partner;
    repOut  << std::setw(13) << replicaData[r].probability
            << std::setw(6) << replicaData[r].switched
            << std::endl;
  }
  DEBUG(2,"replica_exchange_master "<< rank <<":write:\t DONE");

}
  

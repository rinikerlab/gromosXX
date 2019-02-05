/* 
 * File:   replica_exchange_master.cc
 * Author: wissphil, sriniker
 * 
 * Created on April 29, 2011, 2:06 PM
 */

#define REPEX_MPI
#include <util/replicaExchange/replica_exchange_slave.h>

#include <stdheader.h>

#include <algorithm/algorithm.h>
#include <topology/topology.h>
#include <simulation/simulation.h>
#include <configuration/configuration.h>

#include <algorithm/algorithm/algorithm_sequence.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <io/argument.h>
#include <time.h>
#include <unistd.h>

#include <io/configuration/out_configuration.h>

#include <util/replicaExchange/repex_mpi.h>
#include <util/replicaExchange/replica_exchange_base.h>

#ifdef XXMPI
#include <mpi.h>
#endif

util::replica_exchange_slave::replica_exchange_slave(io::Argument & _args,
        int cont,
        int rank,
        std::vector<int> repIDs,
        std::map<ID_t, rank_t> & repMap) : replica_exchange_base(_args, cont, rank, repIDs, repMap) {
}

util::replica_exchange_slave::~replica_exchange_slave() {
}

void util::replica_exchange_slave::send_to_master() const {
#ifdef XXMPI
  DEBUG(2,"replica_exchange_slave " << rank << ":send_to_master \t START");
  for (std::vector<util::replica *>::const_iterator it = replicas.begin(); it < replicas.end(); ++it) {
    util::repInfo info;
    info.run = (*it)->run;
    info.epot = (*it)->epot;
    info.epot_partner = (*it)->epot_partner;
    info.partner = (*it)->partner;
    info.probability = (*it)->probability;
    info.switched = int((*it)->switched);
    MPI_Send(&info, 1, MPI_REPINFO, 0, REPINFO, MPI_COMM_WORLD);
  }
  DEBUG(2,"replica_exchange_slave " << rank << ":\t send_to_master \t Done");
#endif
}

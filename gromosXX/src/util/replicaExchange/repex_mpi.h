/**
 * @file repex_mpi.h
 * some definitions for MPI communication
 */

#ifndef REPEX_MPI_H
#define	REPEX_MPI_H

#ifdef XXMPI
#include <mpi.h>
#endif

#ifdef REPEX_MPI
#define EXTERN
#else
#define EXTERN extern
#endif

#include <util/replicaExchange/replica.h>
#include <util/replicaExchange/replica_reeds.h>

/*
 * define some MPI datatypes
 */
//  EXTERN MPI_Datatype MPI_INITVAL;
#ifdef XXMPI
EXTERN MPI_Datatype MPI_VARRAY;
EXTERN MPI_Datatype MPI_CONFINFO;
EXTERN MPI_Datatype MPI_BOX;
EXTERN MPI_Datatype MPI_REPINFO;
EXTERN MPI_Datatype MPI_EDSINFO;
#endif

/*
 * define message tags for MPI communication
 */
enum {
  INIT, CONF, POS, POSV, VEL, LATTSHIFTS, STOCHINT, BOX, ANGLES, DF, ENERGIES, SWITCHENERGIES, SENDCOORDS, REPINFO, EDSINFO
};

namespace util {

  /**
   * @struct repInfo
   * holds the information that is sent to master after MD run
   */
  struct repInfo {
    int switched;
    int run;
    int partner;
    double epot;
    double epot_partner;
    double probability;
  };
}

/**
 * @typedef ID_t
 * ID type
 */
typedef unsigned int ID_t;
/**
 * @typedef rank_t
 * rank_type
 */
typedef unsigned int rank_t;

#endif	/* REPEX_MPI_H */


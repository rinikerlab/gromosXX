/**
 * @file repex_mpi.cc
 * the main md program for replica exchange simulations using the message passing interface
 */
/**
 * @page programs Program Documentation
 *
 * @anchor repex_mpi
 * @section repex_mpi replica exchange
 * @date 20.08.2019
 *
 * Program repex_mpi is used to run replica exchange simulations.
 *
 * See @ref md for the documentation of the command line arguments.
 * Addition command line arguments are:
 * <table border=0 cellpadding=0>
 * <tr><td> \@repdat</td><td>&lt;name of the replica exchange data file&gt; </td><td style="color:#088A08">in</td></tr>
 * <tr><td> \@repout</td><td>&lt;name of the replica exchange output files&gt; </td><td style="color:#088A08">in</td></tr>
 * </table>
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
#include <util/parse_verbosity.h>
#include <util/usage.h>
#include <util/error.h>

#include <io/read_input.h>
#include <io/parameter/check_parameter.h>
#include <simulation/parameter.h>

#include <io/print_block.h>

#include <thread>
#include <time.h>
#include <unistd.h>

#include <io/configuration/out_configuration.h>
#include <math/gmath.h>


#ifdef XXMPI
#include <mpi.h>
#endif

#include <util/replicaExchange/replica_mpi_tools.h>
#include <util/replicaExchange/replica/replica.h>

#include <util/replicaExchange/replica_exchangers/replica_exchange_master.h>
#include <util/replicaExchange/replica_exchangers/replica_exchange_slave.h>
#include <util/replicaExchange/replica_exchangers/replica_exchange_master_eds.h>
#include <util/replicaExchange/replica_exchangers/replica_exchange_slave_eds.h>

#include <util/replicaExchange/repex_mpi.h>

#undef MODULE
#undef SUBMODULE
#define MODULE util
#define SUBMODULE replica_exchange

int main(int argc, char *argv[]) {
#ifdef XXMPI
    //initializing MPI
    MPI_Init(&argc, &argv);
    const double start = MPI_Wtime();
    
    int ttotalNumberOfThreads;
    int tglobalThreadID;

    MPI_Comm_size(MPI_COMM_WORLD, &ttotalNumberOfThreads);
    MPI_Comm_rank(MPI_COMM_WORLD, &tglobalThreadID);
    
    unsigned int totalNumberOfThreads= ttotalNumberOfThreads;
    unsigned int globalThreadID = tglobalThreadID;
    
    if (globalThreadID == 0) {
        std::string msg("\n==================================================\n\tGROMOS Replica Exchange:\n==================================================\n");
        std::cout << msg;
        std::cerr << msg;
    }

    // reading arguments
    util::Known knowns;
    knowns << "topo" << "conf" << "input" << "verb" << "pttopo"
            << "trc" << "fin" << "trv" << "trf" << "trs" << "tre" << "trg"
            << "bae" << "bag" << "posresspec" << "refpos" << "distrest" << "dihrest"
            << "jval" << "rdc" << "xray" << "lud" << "led" << "print" << "friction" // << "anatrj"
            << "version" << "repdat" << "repout";

    std::string usage;
    util::get_usage(knowns, usage, argv[0]);
    usage += "#\n\n";
    
    // Parse command line arguments
    io::Argument args;
    try {
        if (args.parse(argc, argv, knowns)) {
            std::cerr << usage << std::endl;
            MPI_Finalize();
            return 1;
        }

        if (args.count("version") >= 0) {
            MPI_Finalize();
            return 0;
        }

        // parse the verbosity flag and set debug levels
        if (util::parse_verbosity(args)) {
            std::cerr << "could not parse verbosity argument" << std::endl;
            MPI_Finalize();
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "\n\t########################################################\n"
                << "\n\t\t Exception was thrown in Commandline Parsing!\n"
                << "\n\t########################################################\n";
        std::string msg = "ERROR!\n Uh OH! Caught an Exception in initial Args parse!\n\n";
        std::cout << msg << e.what() << std::endl;
        std::cerr << msg << e.what() << std::endl;
        MPI_Finalize();
        MPI_Abort(MPI_COMM_WORLD, E_USAGE);
        return -1;
    } catch (...) {
        std::cerr << "\n\t########################################################\n"
                << "\n\t\t Exception was thrown in Commandline Parsing!\n"
                << "\n\t########################################################\n";
        std::string msg = "ERROR!\n Uh OH! Caught an non standard Exception in initial Args parse!!\n Hit the developers!\n\n";
        std::cout << msg << std::endl;
        std::cerr << msg << std::endl;
        MPI_Finalize();
        return -1;
    }
    MPI_DEBUG(1, "RANK: "<<globalThreadID<<" totalRankNum: "<< totalNumberOfThreads<<": hello world!\n");

    /**
     * GLOBAL SETTING VARIABLES
     */

    bool reedsSim;    
    unsigned int equil_runs;
    unsigned int sim_runs;
    unsigned int total_runs;
    unsigned int numAtoms;
    unsigned int numReplicas;
    unsigned int numEDSstates;
    unsigned int cont;

    //simulation dependend MPI Vars
    unsigned  subThreadOfSimulation; //MPI thread belongs to replica Simulation:
    unsigned int threadsPerReplicaSimulation; //remove this param - also from exchangers
    unsigned int threadIDinSimulation;
    
    std::map<unsigned int, unsigned int> thread_id_replica_map; // where is which replica
    std::vector<std::vector<unsigned int> > replica_owned_threads; // set IDs for each replica
    
    
    try {
        {
            topology::Topology topo;
            configuration::Configuration conf;
            algorithm::Algorithm_Sequence md;
            simulation::Simulation sim;

            // read in parameters
            bool quiet = true;
            if (globalThreadID == 0) {
                quiet = true;
            }

            if (io::read_parameter(args, sim, std::cout, true)) {
                if (globalThreadID == 0) {
                    std::cerr << "\n\t########################################################\n"
                            << "\n\t\tErrors during read Parameters reading!\n"
                            << "\n\t########################################################\n";
                    io::messages.display(std::cout);
                    io::messages.display(std::cerr);
                }
                MPI_Finalize();
                return 1;
            }
            
            MPI_DEBUG(2, "RANK: "<<globalThreadID<<" done with params\n");    //TODO: lower verb level
            
            //set global parameters
            cont = sim.param().replica.cont;
            equil_runs = sim.param().replica.equilibrate;
            sim_runs = sim.param().replica.trials;
            total_runs = sim.param().replica.trials + equil_runs;
            numAtoms = topo.num_atoms();
            reedsSim = sim.param().reeds.reeds;
            
            if (reedsSim) {
                numReplicas = sim.param().reeds.num_l;
                numEDSstates = sim.param().reeds.eds_para[0].numstates;
            } else {
                numReplicas = sim.param().replica.num_T * sim.param().replica.num_l;
                numEDSstates = 0;
            }
            
            //MPI THREAD SIMULATION SPLITTING
            //needed to be calculated here
            //repIDs every node gets one element of that vector
            replica_owned_threads.resize(numReplicas);

            // counts through every replica and assigns it to respective node
            // starts at beginning if numReplicas > size
            // could be optimized by putting neighboring replicas on same node; less communication...
            //TODO: not sure if this code really does what it should.! (idea would)

            //MPI THREAD SPLITING ONTO Simulation - REPLICAS
            threadsPerReplicaSimulation = totalNumberOfThreads / numReplicas;
            unsigned int leftOverThreads = totalNumberOfThreads % numReplicas;
            
            unsigned int threadID =0;
            int replica_offset = 0;
            std::cout << "left_over "<< leftOverThreads << "\n";
            for (unsigned int replicaSimulationID = 0; replicaSimulationID < numReplicas; replicaSimulationID++) {

                for (unsigned int replicaSimulationSubThread = 0; replicaSimulationSubThread < threadsPerReplicaSimulation; replicaSimulationSubThread++) {

                    threadID = replicaSimulationSubThread + replicaSimulationID*threadsPerReplicaSimulation+replica_offset;
                    thread_id_replica_map.insert(std::pair<unsigned int, unsigned int>(threadID, replicaSimulationID));
                    replica_owned_threads[replicaSimulationID].push_back(threadID);

                }
                if(leftOverThreads>0 and replicaSimulationID < leftOverThreads){    //left over threads are evenly distirbuted.

                    threadID = threadsPerReplicaSimulation + replicaSimulationID*totalNumberOfThreads+replica_offset;
                    thread_id_replica_map.insert(std::pair<unsigned int, unsigned int>(threadID, replicaSimulationID));
                    replica_owned_threads[replicaSimulationID].push_back(threadID);
                    replica_offset++;

                }    
            }            
            
            subThreadOfSimulation = thread_id_replica_map[globalThreadID];
            
            int counter = 0;
            for(unsigned int x : replica_owned_threads[subThreadOfSimulation]){
                if(x==globalThreadID){
                    threadIDinSimulation = counter;
                    break;
                }
                counter++;
            } 
            
            // read in the rest
            if (io::read_input_repex(args, topo, conf, sim, md, subThreadOfSimulation, globalThreadID, std::cout, quiet)) {
                if (globalThreadID == 0) {
                    std::cerr << "\n\t########################################################\n"
                            << "\n\t\tErrors during initial Parameter reading!\n"
                            << "\n\t########################################################\n";
                    io::messages.display(std::cout);
                    io::messages.display(std::cerr);
                    MPI_Abort(MPI_COMM_WORLD, E_USAGE);
                }
                MPI_Finalize();
                return 1;
            }
            MPI_DEBUG(2, "RANK: "<<globalThreadID<<" done with repex_in\n");    //TODO: lower verb level
            
            //if any replica Ex block - present   
            if (sim.param().reeds.reeds == false && sim.param().replica.retl == false) {
                if (globalThreadID == 0) {
                    std::cerr << "\n\t########################################################\n"
                            << "\n\t\tErrors during initial Parameter reading! "
                            << "\n\t\t    No repex block was satisfied!\n"
                            << "\n\t########################################################\n";
                    std::cerr << "\n Please add one RE-block (e.g.:REPLICA or REEDS) to the imd file.\n";
                    std::cout << "\n Please add one RE-block (e.g.:REPLICA or REEDS)  to the imd file.\n";
                }
                MPI_Finalize();
                return 1;
            }
            MPI_DEBUG(2, "RANK: "<<globalThreadID<<" done with replica Ex check\n");    //TODO: lower verb level
            
            if (io::check_parameter(sim)) {
                if (globalThreadID == 0) {
                    std::cerr << "\n\t########################################################\n"
                            << "\n\t\tErrors during initial Parameter reading!\n"
                            << "\n\t########################################################\n";
                    io::messages.display(std::cout);
                    io::messages.display(std::cerr);
                }
                MPI_Finalize();
                return 1;
            }
            MPI_DEBUG(2, "RANK: "<<globalThreadID<<" done with param check\n");    //TODO: lower verb level
            
  
            //SOME additional Checks
            //if enough threads avail
            if (totalNumberOfThreads < numReplicas) {
                if (globalThreadID == 0) {
                    std::cerr << "\n\t########################################################\n"
                            << "\n\t\tErrors during initial Parameter reading!\n"
                            << "\n\t########################################################\n";
                    std::cerr << "\n There were not enough MPI threads assigned to this run!\n"
                            << "FOUND THREADS: " << totalNumberOfThreads << "\tNEED: " << numReplicas << "\n";
                    std::cout << "\n There were not enough MPI thread assigned to this run!\n"
                            << "FOUND THREADS: " << totalNumberOfThreads << "\tNEED: " << numReplicas << "\n";
                    MPI_Finalize();

                }
                return -1;
            }

            if (globalThreadID == 0) { //Nice message
                std::ostringstream msg;
                msg << "\t Short RE-Setting Overview:\n";
                msg << "\t continuation:\t" << cont << "\n";
                msg << "\t equilibration runs:\t" << equil_runs << "\n";
                msg << "\t Exchange Trials runs:\t" << sim_runs << "\n";
                msg << "\t Simulation Steps Between Trials:\t" << sim.param().step.number_of_steps << "\n";
                msg << "\t Total Simulation Time:\t" << sim.param().step.number_of_steps * sim_runs * sim.param().step.dt << "ps\n";
                msg << "\t numReplicas:\t" << numReplicas << "\n";

                if (reedsSim) {
                    msg << "\n\t RE-EDS:\n";
                    msg << "\t numSValues:\t" << sim.param().reeds.num_l << "\n";
                    msg << "\t numStates:\t" << numEDSstates << "\n";
                }

                msg << "\n==================================================\n\tFinished Initial Parsing\n\n==================================================\n";
                std::cout << msg.str();
                std::cerr << msg.str();

            }
        }
    } catch (const std::exception &e) {
        std::cerr << "\n\t########################################################\n"
                << "\n\t\t Exception was thrown in initial parsing!\n"
                << "\n\t########################################################\n";
        std::string msg = "ERROR!\n Uh OH! Caught an Exception in initial test Parsing of the Gromos Parameters!\n\nMessage:\t";
        std::cout << msg << e.what() << std::endl;
        std::cerr << msg << e.what() << std::endl;
        MPI_Finalize();
        return -1;
    } catch (...) {
        std::cerr << "\n\t########################################################\n"
                << "\n\t\t Exception was thrown!\n"
                << "\n\t########################################################\n";
        std::string msg = "ERROR!\n Uh OH! Caught an non standard Exception in initial test Parsing of the  Gromos Parameters!\n Hit the developers!\n\nMessage:\t";
        std::cout << msg << std::endl;
        std::cerr << msg << std::endl;
        MPI_Finalize();
        return -1;
    }
    MPI_DEBUG(1, "RANK: "<<globalThreadID<<" Done with init parse\n");    //TODO: lower verb level
   
    io::messages.clear();
    MPI_DEBUG(1, "REPLICA_ID \t " << globalThreadID << "\t Simulation_ID\t"<< subThreadOfSimulation << "\t SimulationThread\t"<<threadIDinSimulation<<"\n")

    /**
     *  MPI PARAMETERS
     */

    //GLOBAL MPI VARIABLES:
    MPI_Datatype MPI_VEC;
    simulation::mpi_control_struct tmp_mpi = simulation::mpi_control_struct();
    //global number of threads and unique global threadID
    tmp_mpi.global_number_of_threads = totalNumberOfThreads;
    tmp_mpi.global_thread_id = globalThreadID;
    //local threads
    tmp_mpi.local_simulation = subThreadOfSimulation;
    tmp_mpi.local_number_of_threads = threadsPerReplicaSimulation;
    tmp_mpi.local_master = replica_owned_threads[thread_id_replica_map[globalThreadID]][0];
    tmp_mpi.local_thread_id = threadIDinSimulation;
    //maps and structures to find everything
    tmp_mpi.thread_id_replica_map = thread_id_replica_map;
    tmp_mpi.replica_owned_threads = replica_owned_threads;   

    try { //Exchange structures
        // Vector
        MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VEC);
        MPI_Type_commit(&MPI_VEC);

        // Box
        MPI_Type_contiguous(3, MPI_VEC, &MPI_BOX);
        MPI_Type_commit(&MPI_BOX);

        // VArray with size of system
        MPI_Type_contiguous(numAtoms, MPI_VEC, &MPI_VARRAY);
        MPI_Type_commit(&MPI_VARRAY);

        // defining struct with non static replica information
        int blocklen[] = {3, 3};
        MPI_Datatype typ[] = {MPI_INT, MPI_DOUBLE};
        MPI_Aint intext;
        MPI_Aint lb; //not used
        MPI_Type_get_extent(MPI_INT, &lb, &intext);

        MPI_Aint disps[] = {(MPI_Aint) 0, 4 * intext};
        MPI_Type_create_struct(2, blocklen, disps, typ, &MPI_REPINFO);
        MPI_Type_commit(&MPI_REPINFO);

        if (reedsSim) {
            MPI_Type_contiguous(numEDSstates, MPI_DOUBLE, &MPI_EDSINFO);
            MPI_Type_commit(&MPI_EDSINFO);
        }
    } catch (const std::exception &e) {
        std::cerr << "\n\t########################################################\n"
                << "\n\t\tErrors during MPI-Initialisation!\n"
                << "\n\t########################################################\n";
        std::string msg = "ERROR!\n Uh OH! Caught an Exception in MPI-Initialisation!\n\nMessage:\t";
        std::cout << msg << e.what() << std::endl;
        std::cerr << msg << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n\t########################################################\n"
                << "\n\t\tErrors during MPI-Initialisation!\n"
                << "\n\t########################################################\n";
        std::string msg = "ERROR!\n Uh OH! Caught an non standard Exception in initial test Parsing of the MPI Parameters!\n Hit the developers!\n\nMessage:\t";
        std::cout << msg << std::endl;
        std::cerr << msg << std::endl;
        MPI_Finalize();
        return -1;
    }

    // make sure all nodes have initialized everything
    if(globalThreadID==0){  //if DEBUG TODO: remove bschroed
        std::cout<< "repIDS:\t\n";
        for(unsigned int x=0; x < tmp_mpi.replica_owned_threads.size(); x++){
            std::cout << "\t" << x << ". Replica: ";
            for(int threadID :  tmp_mpi.replica_owned_threads[x]){
                std::cout << "\t" << threadID;
            }
            std::cout<< "\n";
        }
        std::cout<< "\n";
        std::cout<< "repMap:\n";
        for(std::pair<unsigned int, unsigned int> p : tmp_mpi.thread_id_replica_map){
            std::cout << "\t " << p.first << "\t" << p.second <<"\n";
        }
        std::cout<< "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(tmp_mpi.local_number_of_threads > 1){
        throw "using more than 1 thread per replica is currently not implemented!";
    }
    
    
    
    //////////////////////////////
    /// Starting master-slave Pattern
    //////////////////////////////

    if (globalThreadID == 0) { //MASTER
        //std::cout << "RANK: "<< uniqueThreadID <<"\tSLEPPING Master\n";
        //MPI_Barrier(MPI_COMM_WORLD);
                
        //nice messages
        std::cout << "\n==================================================\n\tStart REPLICA EXCHANGE SIMULATION:\n\n==================================================\n";
        std::cout << "numreplicas:\t " << numReplicas << "\n";
        std::cout << "num Slaves:\t " << numReplicas - 1 << "\n";
        std::cerr << "\n==================================================\n\tStart REPLICA EXCHANGE SIMULATION:\n\n==================================================\n";
        std::cerr << "numreplicas:\t " << numReplicas << "\n";
        std::cerr << "num Slaves:\t " << numReplicas - 1 << "\n";
        MPI_DEBUG(1, "Master \t " << globalThreadID<< "simulation: " << subThreadOfSimulation)

        // Select repex Implementation - Polymorphism
        util::replica_exchange_master* Master;
        try {
            if (reedsSim) {
                DEBUG(1, "Master_eds \t Constructor")    
                Master = new util::replica_exchange_master_eds(args, cont, globalThreadID, replica_owned_threads, thread_id_replica_map);
            } else {
                DEBUG(1, "Master \t Constructor")
                Master = new util::replica_exchange_master(args, cont, globalThreadID, replica_owned_threads, thread_id_replica_map);
            }
        } catch (...) {
            std::cerr << "\n\t########################################################\n"
                    << "\n\t\tErrors during Master Class Init!\n"
                    << "\n\t########################################################\n";
            std::cerr << "\nREPEX:       Failed in initialization of Replica rank: " << globalThreadID << "" << std::endl;
            std::cout << "\nREPEX:       Failed in initialization of Replica rank: " << globalThreadID << "" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, E_USAGE);
            return 1;
        }

        MPI_DEBUG(1, "Master \t INIT")
        Master->init();
        Master->init_repOut_stat_file();

        //do md:
        unsigned int trial = 0;
        MPI_DEBUG(1, "Master \t \t \t Equil: " << equil_runs)
        for (; trial < equil_runs; ++trial) { // for equilibrations
            Master->run_MD();
        }
        MPI_DEBUG(1, "Master \t \t MD: " << total_runs)

                //Vars for timing
                int hh, mm, ss, eta_hh, eta_mm, eta_ss = 0;
        double eta_spent, percent, spent = 0.0;
        trial = 0; //reset trials
        for (; trial < sim_runs; ++trial) { //for repex execution
            DEBUG(2, "Master " << globalThreadID << " \t MD trial: " << trial << "\n")\
            DEBUG(2, "Master " << globalThreadID << " \t run_MD START " << trial << "\n")
            Master->run_MD();
            DEBUG(2, "Master " << globalThreadID << " \t run_MD DONE " << trial << "\n")

            DEBUG(2, "Master " << globalThreadID << " \t swap START " << trial << "\n")
            Master->swap();
            DEBUG(2, "Master " << globalThreadID << " \t run_MD DONE " << trial << "\n")

            DEBUG(2, "Master " << globalThreadID << " \t receive START " << trial << "\n")
            Master->receive_from_all_slaves();
            DEBUG(2, "Master " << globalThreadID << " \t write START " << trial << "\n")
            Master->write();

            if ((total_runs / 10 > 0) && (trial % (total_runs / 10) == 0)) { //Timer 
                percent = double(trial) / double(total_runs);
                spent = util::now() - start;
                hh = int(spent / 3600);
                mm = int((spent - hh * 3600) / 60);
                ss = int(spent - hh * 3600 - mm * 60);

                std::cout << "\nREPEX:       " << std::setw(2) << percent * 100 << "% done..." << std::endl;
                std::cout << "REPEX: spent " << hh << ":" << mm << ":" << ss << std::endl;
                std::cerr << "\nREPEX:       " << std::setw(2) << percent * 100 << "% done..." << std::endl;
                std::cerr << "REPEX: spent " << hh << ":" << mm << ":" << ss << std::endl;
            }
        }
        MPI_DEBUG(1, "Master \t \t finalize ")

        Master->write_final_conf();
        std::cout << "\n=================== Master Node " << globalThreadID << "  finished successfully!\n";

        } else { //SLAVES    
        MPI_DEBUG(1, "Slave " << globalThreadID << "simulation: " << subThreadOfSimulation)

        // Select repex Implementation - Polymorphism
        util::replica_exchange_slave* Slave;
        if (reedsSim) {
            Slave = new util::replica_exchange_slave_eds(args, cont, globalThreadID, replica_owned_threads, thread_id_replica_map);
        } else {
            Slave = new util::replica_exchange_slave(args, cont, globalThreadID, replica_owned_threads, thread_id_replica_map);
        }

        MPI_DEBUG(1, "Slave " << globalThreadID << " \t INIT")
        Slave->init();

        //do md:
        unsigned int trial = 0;
        MPI_DEBUG(1, "Slave " << globalThreadID << " \t EQUIL " << equil_runs << " steps")
        for (; trial < equil_runs; ++trial) { // for equilibrations
            Slave->run_MD();
        }

        MPI_DEBUG(1, "Slave " << globalThreadID << " \t MD " << total_runs << " steps")
        for (; trial < total_runs; ++trial) { //for repex execution
            DEBUG(2, "Slave " << globalThreadID << " \t MD trial: " << trial << "\n")
            DEBUG(2, "Slave " << globalThreadID << " \t run_MD START " << trial << "\n")
            Slave->run_MD();
            DEBUG(2, "Slave " << globalThreadID << " \t swap START " << trial << "\n")
            Slave->swap();
            DEBUG(2, "Slave " << globalThreadID << " \t send START " << trial << "\n")
            Slave->send_to_master();
        }

        MPI_DEBUG(1, "Slave " << globalThreadID << " \t Finalize")
        
        Slave->write_final_conf();
        std::cout << "\n=================== Slave Node " << globalThreadID << "  finished successfully!\n";
    }

    MPI_Barrier(MPI_COMM_WORLD); //Make sure all processes finished.

    //last MASTER OUTPUT
    if (globalThreadID == 0) {
        //FINAL OUTPUT - Time used:
        double end = MPI_Wtime();
        double duration = end - start;
        int durationMin = int(duration / 60);
        int durationHour = int(std::floor(durationMin / 60));
        int durationMinlHour = int(std::floor(durationMin - durationHour * 60));
        int durationSlMin = int(std::floor(duration - (durationMinlHour + durationHour * 60)*60));

        //Todo: if (cond){finished succ}else{not} bschroed
        std::string msg("\n====================================================================================================\n\tREPLICA EXCHANGE SIMULATION finished!\n====================================================================================================\n");
        std::cout << msg;
        std::cerr << msg;

        std::cout << "TOTAL TIME USED: \n\th:min:s\t\tseconds\n"
                << "\t" << durationHour << ":" << durationMinlHour << ":" << durationSlMin << "\t\t" << duration << "\n";
        std::cerr << "TOTAL TIME USED: \n\th:min:s\t\tseconds\n"
                << "\t" << durationHour << ":" << durationMinlHour << ":" << durationSlMin << "\t\t" << duration << "\n";
    }

    MPI_Finalize();
    return 0;
#else
    std::cout << "\n==================================================\n\tGROMOS Replica Exchange:\n==================================================\n";
    std::cout << argv[0] << " needs MPI to run\n\tuse --enable-mpi at configure and appropriate compilers\n" << std::endl;
    return 1;
#endif
}


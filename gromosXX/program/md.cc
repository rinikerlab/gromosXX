
#include <util/stdheader.h>

#include <topology/core/core.h>
#include <topology/topology.h>
#include <simulation/multibath.h>
#include <simulation/parameter.h>
#include <simulation/simulation.h>
#include <configuration/energy.h>
#include <configuration/energy_average.h>
#include <configuration/configuration.h>
#include <algorithm/algorithm.h>
#include <algorithm/algorithm/algorithm_sequence.h>
#include <interaction/interaction.h>
#include <interaction/forcefield/forcefield.h>

#include <io/argument.h>
#include <util/parse_verbosity.h>
#include <io/read_input.h>

#include <io/configuration/out_configuration.h>

int main(int argc, char *argv[]){
  try{
    
    char *knowns[] = 
      {
        "topo", "conf", "input", "verb", "alg", "pttopo",
        "trj", "fin", "trv", "trf", "tre", "trg", "print", "trp"
      };
    
    int nknowns = 14;
    
    std::string usage = argv[0];
    usage += "\n\t@topo    <topology>\n";
    usage += "\t[@pttopo <perturbation topology>]\n";
    usage += "\t@conf    <starting configuration>\n";
    usage += "\t@input   <input>\n";
    usage += "\t@trj     <trajectory>\n";
    usage += "\t@fin     <final structure>\n";
    usage += "\t[@trv    <velocity trajectory>]\n";
    usage += "\t[@trf    <force trajectory>]\n";
    usage += "\t[@tre    <energy trajectory>]\n";
    usage += "\t[@trg    <free energy trajectory>]\n";
    usage += "\t[@alg    <RK|LF>]\n";
    usage += "\t[@print  <pairlist/force>]\n";
    usage += "\t[@trp    <print file>]\n";
    usage += "\t[@verb   <[module:][submodule:]level>]\n";

    io::Argument args(argc, argv, nknowns, knowns, usage);

    // parse the verbosity flag and set debug levels
    util::parse_verbosity(args);

    // create the simulation classes
    simulation::Parameter param;
    topology::Topology topo;
    configuration::Configuration conf;
    algorithm::Algorithm_Sequence md;
    
    io::read_input(args, param, topo, conf, md);
    simulation::Simulation sim(param);

    io::Out_Configuration traj("GromosXX");

    if (args.count("fin") > 0)
      traj.final_configuration(args["fin"]);
    else throw std::string("argument fin for final configuration required!");
    if (args.count("trj") > 0)
      traj.trajectory(args["trj"], param.write.position);
    else if (param.write.position)
      throw std::string("write trajectory but no trj argument");
    if (args.count("trv") > 0)
      traj.velocity_trajectory(args["trv"], param.write.velocity);
    else if (param.write.velocity)
      throw std::string("write velocity trajectory but no trv argument");
    if (args.count("trf") > 0)
      traj.force_trajectory(args["trf"], 1);
    //else if (param.write.force)
    //  throw std::string("write force trajectory but no trf argument");
    if (args.count("tre") > 0)
      traj.energy_trajectory(args["tre"], param.write.energy);
    else if (param.write.energy)
      throw std::string("write energy trajectory but no tre argument");
    if (args.count("trg") > 0)
      traj.free_energy_trajectory(args["trg"], param.write.free_energy);
    else if (param.write.free_energy)
      throw std::string("write free energy trajectory but no trg argument");

    std::cout << "\nMESSAGES FROM INITIALIZATION\n";
    io::messages.display(std::cout);
    io::messages.clear();

    std::cout << "\nenter the next level of molecular "
	      << "dynamics simulations\n" << std::endl;

    double end_time = sim.param().step.t0 + 
      sim.time_step_size() * sim.param().step.number_of_steps;
    

    std::cout << "MD loop\n\tstart t=" << sim.time() 
	      << "\tend t=" << end_time << std::endl;
    
    while(sim.time() < end_time){
      std::cout << "md step " << sim.time() << std::endl;
      
      traj.write(conf, topo, sim, io::reduced);

      md.run(topo, conf, sim);

      sim.time() += sim.time_step_size();
      ++sim.steps();
      
    }
    

    std::cout << "writing final configuration" << std::endl;
    
    traj.write(conf, topo, sim, io::final);
    
    std::cout << "\nMESSAGES FROM SIMULATION\n";
    io::messages.display(std::cout);

    std::cout << "\nGromosXX finished successfully\n" << std::endl;
    
  }
  catch (std::string s){
    io::messages.display();
    std::cerr << "there was something wrong:\n" << s << std::endl;
    return 1;
  }
  
    return 0;
}


/**
 * @file standard_pairlist_algorithm.h
 * standard pairlist algorithm (reference implementation)
 */

#ifndef INCLUDED_STANDARD_PAIRLIST_ALGORITHM_H
#define INCLUDED_STANDARD_PAIRLIST_ALGORITHM_H

namespace math
{
  template<math::boundary_enum>
  class Periodicity;
}


namespace interaction
{
  class Storage;
  class Pairlist;
 
  template<typename t_interaction_spec>
  class Nonbonded_Innerloop;
  
  template<typename t_interaction_spec, typename t_perturbation_details>
  class Perturbed_Nonbonded_Innerloop;
  
  /**
   * @class Standard_Pairlist_Algorithm
   * create an atomic pairlist with a
   * chargegroup based or atom based
   *  cut-off criterion.
   */
  class Standard_Pairlist_Algorithm : 
    public Pairlist_Algorithm
  {
  public:
    /**
     * Constructor.
     */
    Standard_Pairlist_Algorithm();

    /**
     * Destructor.
     */
    virtual ~Standard_Pairlist_Algorithm() {}

    /**
     * init pairlist
     */
    int init(Nonbonded_Parameter * param);
    
    /**
     * prepare the pairlists
     */    
    virtual void prepare(topology::Topology & topo,
			 configuration::Configuration & conf,
			 simulation::Simulation &sim);

    /**
     * update the pairlist
     */
    virtual void update(topology::Topology & topo,
			configuration::Configuration & conf,
			simulation::Simulation & sim,
			interaction::Storage & storage,
			interaction::Pairlist & pairlist,
			unsigned int begin, unsigned int end,
			unsigned int stride);

    /**
     * update the pairlist, separating perturbed and nonperturbed interactions
     */
    virtual void update_perturbed(topology::Topology & topo,
				  configuration::Configuration & conf,
				  simulation::Simulation & sim,
				  interaction::Storage & storage,
				  interaction::Pairlist & pairlist,
				  interaction::Pairlist & perturbed_pairlist,
				  unsigned int begin, unsigned int end,
				  unsigned int stride);
        
  protected:

    void update_cg(topology::Topology & topo,
		   configuration::Configuration & conf,
		   simulation::Simulation & sim,
		   interaction::Storage & storage,
		   interaction::Pairlist & pairlist,
		   unsigned int begin, unsigned int end,
		   unsigned int stride);

    template<typename t_perturbation_details>
    void update_pert_cg(topology::Topology & topo,
			configuration::Configuration & conf,
			simulation::Simulation & sim,
			interaction::Storage & storage,
			interaction::Pairlist & pairlist,
			interaction::Pairlist & perturbed_pairlist,
			unsigned int begin, unsigned int end,
			unsigned int stride);

    template<typename t_interaction_spec>
    void _update_cg(topology::Topology & topo,
		    configuration::Configuration & conf,
		    simulation::Simulation & sim, 
		    interaction::Storage & storage,
		    interaction::Pairlist & pairlist,
		    unsigned int begin, unsigned int end,
		    unsigned int stride);

    template<typename t_interaction_spec, typename t_perturbation_details>
    void _update_pert_cg(topology::Topology & topo,
			 configuration::Configuration & conf,
			 simulation::Simulation & sim, 
			 interaction::Storage & storage,
			 interaction::Pairlist & pairlist,
			 interaction::Pairlist & perturbed_pairlist,
			 unsigned int begin, unsigned int end,
			 unsigned int stride);

    template<typename t_interaction_spec>
    void do_cg1_loop(topology::Topology & topo,
		     configuration::Configuration & conf,
		     interaction::Storage & storage,
		     interaction::Pairlist & pairlist,
		     Nonbonded_Innerloop<t_interaction_spec> & innerloop,
		     topology::Chargegroup_Iterator const & cg1,
		     int cg1_index, int num_solute_cg, int num_cg,
		     math::Periodicity<t_interaction_spec::boundary_type> const & periodicity);

    template<typename t_interaction_spec, typename t_perturbation_details>
    void do_pert_cg1_loop(topology::Topology & topo,
			  configuration::Configuration & conf,
			  interaction::Storage & storage,
			  interaction::Pairlist & pairlist,
			  interaction::Pairlist & perturbed_pairlist,
			  Nonbonded_Innerloop<t_interaction_spec> & innerloop,
			  Perturbed_Nonbonded_Innerloop
			  <t_interaction_spec, t_perturbation_details>
			  & perturbed_innerloop,
			  topology::Chargegroup_Iterator const & cg1,
			  int cg1_index, int num_solute_cg, int num_cg,
			  math::Periodicity<t_interaction_spec::boundary_type>
			  const & periodicity);
    
    void do_cg_interaction(topology::Chargegroup_Iterator const &cg1,
			   topology::Chargegroup_Iterator const &cg2,
			   interaction::Pairlist & pairlist);

    void do_pert_cg_interaction(topology::Topology & topo,
				topology::Chargegroup_Iterator const &cg1,
				topology::Chargegroup_Iterator const &cg2,
				interaction::Pairlist & pairlist,
				interaction::Pairlist & perturbed_pairlist);
    
    void do_cg_interaction_excl(topology::Topology & topo,
				topology::Chargegroup_Iterator const &cg1,
				topology::Chargegroup_Iterator const &cg2,
				interaction::Pairlist & pairlist);

    void do_pert_cg_interaction_excl(topology::Topology & topo,
				     topology::Chargegroup_Iterator const &cg1,
				     topology::Chargegroup_Iterator const &cg2,
				     interaction::Pairlist & pairlist,
				     interaction::Pairlist & perturbed_pairlist);
    
    void do_cg_interaction_intra(topology::Topology & topo,
				 topology::Chargegroup_Iterator const &cg1,
				 interaction::Pairlist & pairlist);

    void do_pert_cg_interaction_intra(topology::Topology & topo,
				      topology::Chargegroup_Iterator const &cg1,
				      interaction::Pairlist & pairlist,
				      interaction::Pairlist & perturbed_pairlist);

    void update_atomic(topology::Topology & topo,
		       configuration::Configuration & conf,
		       simulation::Simulation & sim,
		       interaction::Storage & storage,
		       interaction::Pairlist & pairlist,
		       unsigned int begin, unsigned int end,
		       unsigned int stride);

    template<typename t_interaction_spec>
    void _update_atomic(topology::Topology & topo,
			configuration::Configuration & conf,
			simulation::Simulation & sim, 
			interaction::Storage & storage,
			interaction::Pairlist & pairlist,
			unsigned int begin, unsigned int end,
			unsigned int stride);
    
    template<typename t_perturbation_details>
    void update_pert_atomic(topology::Topology & topo,
			    configuration::Configuration & conf,
			    simulation::Simulation & sim,
			    interaction::Storage & storage,
			    interaction::Pairlist & pairlist,
			    interaction::Pairlist & perturbed_pairlist,
			    unsigned int begin, unsigned int end,
			    unsigned int stride);

    template<typename t_interaction_spec, typename t_perturbation_details>
    void _update_pert_atomic(topology::Topology & topo,
			     configuration::Configuration & conf,
			     simulation::Simulation & sim, 
			     interaction::Storage & storage,
			     interaction::Pairlist & pairlist,
			     interaction::Pairlist & perturbed_pairlist,
			     unsigned int begin, unsigned int end,
			     unsigned int stride);

    bool excluded_solute_pair(topology::Topology & topo,
			      unsigned int i, unsigned int j);

    void set_cutoff(double const cutoff_short, double const cutoff_long)
    {
      m_cutoff_long = cutoff_long;
      m_cutoff_short = cutoff_short;
      m_cutoff_short_2 = cutoff_short * cutoff_short;
      m_cutoff_long_2  = cutoff_long * cutoff_long;
    }

  private:
    /**
     * the chargegroup center of geometries.
     */
    math::VArray m_cg_cog;
    /**
     * squared shortrange cutoff.
     */
    double m_cutoff_short_2;
    /**
     * squared longrange cutoff.
     */
    double m_cutoff_long_2;
    /**
     * longrange cutoff.
     */
    double m_cutoff_long;
    /**
     * shortrange cutoff.
     */
    double m_cutoff_short;

  };
} // interaction

#endif

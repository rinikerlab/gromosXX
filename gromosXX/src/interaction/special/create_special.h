/**
 * @file create_special.h
 * create the bonded terms.
 */

#ifndef INCLUDED_CREATE_SPECIAL_H
#define INCLUDED_CREATE_SPECIAL_H

namespace interaction
{
  int create_special(interaction::Forcefield & ff,
		     topology::Topology const & topo,
		     simulation::Parameter const & param,
		     std::ostream & os = std::cout,
		     bool quiet = false);

#ifdef TORCH
/**
 * Add a new Torch model to the Force Field, templated to take in precision
 */
	template<template<typename> typename T>
	void add_torch_model(const simulation::torch_model& model, interaction::Forcefield& ff) {
        if (model.precision == torch::kFloat16) {
          ff.push_back(new T<torch::Half>(model));
        }
        else if (model.precision == torch::kFloat32) {
          ff.push_back(new T<float>(model));
        }
        else {
          ff.push_back(new T<double>(model));
        }
  }
#endif
}

#endif

/**
 * @file periodicity.h
 * periodic boundary conditions (triclinic)
 */

#ifndef INCLUDED_PERIODICITY_H
#define INCLUDED_PERIODICITY_H

#include "math.h"

namespace math
{

  /**
   * @class Periodicity
   * the periodic boundary condition functions.
   */
  template<boundary_enum b>
  class Periodicity : public Boundary_Implementation<b>
  {
  public:
    Periodicity(boundary_enum boundary = b);
    void put_into_box(Vec &v)const;
    void put_into_positive_box(Vec &v)const;
  private:
  };
  
} // math

// inline functions
#include "periodicity.tcc"

#endif
  
  
    

/**
 * @file boundary_implementation.h
 * periodic boundary conditions (triclinic)
 * nearest image implementation.
 */

#ifndef INCLUDED_BOUNDARY_IMPLEMENTATION_H
#define INCLUDED_BOUNDARY_IMPLEMENTATION_H

namespace math
{
  /**
   * @enum boundary_enum
   * boundary condition
   */
  enum boundary_enum{
    /**
     * vacuum.
     */
    vacuum = 0,
    /**
     * triclinic box
     */
    triclinic = 1,
    /**
     * non-specialized version
     */
    any = 2
  };

  /**
   * @class Boundary_Implementation
   * implements the specific functions of
   * the Periodicity class.
   */
  template<boundary_enum b>
  class Boundary_Implementation
  {
  public:
    static int const K = 0;
    static int const L = 1;
    static int const M = 2;
    
    /**
     * Constructor.
     * @param boundary is the boundary condition.
     */
    Boundary_Implementation(boundary_enum boundary = b);
    /**
     * Get the nearest image of v1 in respect to v2 (v1 - v2).
     */
    void nearest_image(Vec const &v1, Vec const &v2, Vec &nim)const;
    /**
     * set the boundary condition.
     */
    void boundary_condition(boundary_enum const b);
    /**
     * get the boundary condition.
     */
    boundary_enum const boundary_condition()const;
    /**
     * get the box.
     */
    Matrix const & box()const;
    /**
     * get a box vector.
     */
    // Vec const & box(size_t const d)const;
    /**
     * get a box element.
     */
    double const box(size_t const d1, size_t const d2)const;

    /**
     * set the box.
     */
    void box(Matrix const &m);
    // not implemented
    /**
     * set the box.
     */
    // void box(Vec v1, Vec v2, Vec v3);
    
  protected:
    /**
     * reference to the system::box.
     */
    Matrix m_box;
    /**
     * the boundary condition.
     */
    boundary_enum m_boundary;
    /**
     * the box volume.
     */
    double m_volume;
    /**
     * triclinic nearest image:
     * -(L*M) / vol
     * -(K*M) / vol
     * -(K*L) / vol
     */
    Matrix m_cross_K_L_M;

  };
  
  /**
   * @class Boundary_Implementation<vacuum>
   * Specialized version for vacuum.
   */
  template<>
  class Boundary_Implementation<vacuum>
  {
  public:
    static int const K = 0;
    static int const L = 1;
    static int const M = 2;
    /**
     * Constructor.
     * @param boundary is the boundary condition.
     */
    Boundary_Implementation(boundary_enum boundary = vacuum);
    /**
     * Get the nearest image of v1 in respect to v2 (v1 - v2).
     */
    void nearest_image(Vec const &v1, Vec const &v2, Vec &nim)const;
    /**
     * set the boundary condition.
     */
    void boundary_condition(boundary_enum const b);
    /**
     * get the boundary condition.
     */
    boundary_enum const boundary_condition()const;
    /**
     * get the box.
     */
    Matrix const box()const;
    /**
     * get a box vector.
     */
    // Vec const box(size_t const d)const;
    /**
     * get a box element.
     */
    double const box(size_t const d1, size_t const d2)const;

    /**
     * set the box.
     */
    void box(Matrix const &m);
    // not implemented
    /**
     * set the box.
     */
    // void box(Vec v1, Vec v2, Vec v3);

  protected:
    /**
     * reference to the system::box.
     */
    Matrix m_box;
    /**
     * the box volume.
     */
    double m_volume;
  };
  
  /**
   * @class Boundary_Implementation<triclinic>
   * specialized version for triclinic boundary conditions.
   */
  template<>
  class Boundary_Implementation<triclinic>
  {
  public:
    static int const K = 0;
    static int const L = 1;
    static int const M = 2;
    /**
     * Constructor.
     * @param boundary is the boundary condition.
     */
    Boundary_Implementation(boundary_enum boundary = triclinic);
    /**
     * Get the nearest image of v1 in respect to v2 (v1 - v2).
     */
    void nearest_image(Vec const &v1, Vec const &v2, Vec &nim)const;
    /**
     * set the boundary condition.
     */
    void boundary_condition(boundary_enum const b);
    /**
     * get the boundary condition.
     */
    boundary_enum const boundary_condition()const;

    /**
     * get the box.
     */
    Matrix const & box()const;
    /**
     * get a box vector.
     */
    // Vec const & box(size_t const d)const;
    /**
     * get a box element.
     */
    double const box(size_t const d1, size_t const d2)const;

    /**
     * set the box.
     */
    void box(Matrix const &m);
    // not implemented
    /**
     * set the box.
     */
    // void box(Vec v1, Vec v2, Vec v3);

  protected:
    /**
     * reference to the system::box.
     */
    Matrix m_box;
    /**
     * the box volume.
     */
    double m_volume;
    /**
     * triclinic nearest image:
     * -(L*M) / vol
     * -(K*M) / vol
     * -(K*L) / vol
     */
    Matrix m_cross_K_L_M;

  };
  
}

#include "boundary_implementation.tcc"

#endif

/**
 * @file volume.h
 * calculate the volume.
 */

#ifndef INCLUDED_VOLUME_H
#define INCLUDED_VOLUME_H

namespace math
{
  /**
   * calculate the volume.
   */
  double volume(math::Box & box, math::boundary_enum b);
}

#endif

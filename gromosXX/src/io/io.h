/**
 * @file io.h
 * gathers common include directives for io
 */

#include "argument.h"
#include "blockinput.h"
#include "message.h"
#include "GInStream.h"
#include "topology/InTopology.h"
#include "trajectory/InTrajectory.h"
#include "trajectory/OutTrajectory.h"
#include "input/InInput.h"

#ifndef NDEBUG

/**
 * @namespace io
 * provide the input/output routines.
 */
namespace io
{
  /**
   * the module debug level.
   */
  extern int debug_level;
  /**
   * debug level for the submodule trajectory.
   */
  extern int trajectory_debug_level;
  /**
   * debug level for the submodule input.
   */
  extern int input_debug_level;
}

#endif

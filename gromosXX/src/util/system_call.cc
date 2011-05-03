#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include "system_call.h"

#include "../../config.h"

std::string util::default_filename_argument;

int util::system_call(const std::string & command,
          std::string & input_file,
          std::string & output_file) {
  std::ostringstream command_to_launch;
  command_to_launch << command;
  if (!input_file.empty()) {
    command_to_launch << " < " << input_file;
  }
  if (output_file.empty()) {
#ifdef HAVE_TMPNAM
    char tmp[TMP_MAX];
    tmpnam(tmp);
    output_file = std::string(tmp);
#else
    throw std::runtime_error("tmpnam is not available on this platform provide "
            "the name for temporary file to system_call")
#endif
  }
  command_to_launch << " > " << output_file;

#ifdef HAVE_SYSTEM
  return system(command_to_launch.str().c_str());
#else
  throw std::runtime_error("System call is not implemented for this platform.")
#endif
}
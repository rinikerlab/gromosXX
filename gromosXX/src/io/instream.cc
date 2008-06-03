/**
 * @file instream.cc
 * basic input stream class definition.
 */

#include <stdheader.h>
#include <vector>

#include "blockinput.h"
#include "instream.h"
#include "message.h"

#undef MODULE
#undef SUBMODULE
#define MODULE io
#define SUBMODULE topology

void io::GInStream::readTitle() {

  std::vector<std::string> _b;

  io::getblock(*_is, _b);
  if (_b[0] != "TITLE")
    io::messages.add("title block expected: found " + _b[0],
		     "instream",
		     io::message::error);
  title = io::concatenate(_b.begin() + 1, _b.end() - 1, title);
}

void io::GInStream::readStream() {
  std::vector<std::string> buffer;
  while (!stream().eof()) {

    if (!io::getblock(stream(), buffer)) {
      if (buffer.size() && buffer[0] != "") {
        std::cerr << "invalid block " + buffer[0] << " in input file?"
                << std::endl;
      }
      break;
    }

    trimblock(buffer);

    // empty blocks may cause problems
    if (buffer.size() == 2) {
      std::ostringstream out;
      out << "empty block (" << buffer[0] << ")";
      io::messages.add("GInStream", out.str(), io::message::error);
    } else {
      std::string n = buffer[0];
      DEBUG(10, "reading block -" << buffer[0] << "- size " << buffer.size());
      m_block[buffer[0]] = buffer;
    }
    buffer.clear();
  }
}


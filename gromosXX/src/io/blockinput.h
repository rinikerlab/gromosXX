/**
 * @file blockinput.h
 * read in blocks.
 */

#ifndef INCLUDED_BLOCKINPUT_H
#define INCLUDED_BLOCKINPUT_H

namespace io {

  /*
   * The function io::getline provides an override of 
   * std::getline in that it retrieves the next line (separated 
   * by const char& sep) from the input stream, which, after 
   * having been stripped of comments (indicated by 
   * const char& comm), is not empty.
   */
  std::istream& getline(
			       std::istream& is, 
			       std::string& s, 
			       const char& sep = '\n',
			       const char& comm = '#'
			       );

  /*
   * The function io::getblock retrieves the next block  
   * (separated by const string& sep) using io::getline.
   *
   * It throws a runtime_error if the stream's good bit is unset 
   * before it's finished reading a block.
   *
   * Finally, the vector<string> it writes to is resized to the 
   * number of strings read.
   */
  std::istream& getblock(
				std::istream& is, 
				std::vector<std::string>& b, 
				const std::string& sep = "END"
				);

  /*
   * The io::concatenate utility allows the concatenation
   * of entries in a vector<string> into just one string. The
   * resulting entries are separated by const char& sep 
   * (defaulting to a newline character).
   */
  std::string& concatenate(
			   std::vector<std::string>::const_iterator begin,
			   std::vector<std::string>::const_iterator end,
			   std::string& s,
			   const char& sep = '\n'
			   );

  /**
   * trim away leading empty lines (only whitespace)
   */
  void
  trimblock(std::vector<std::string> &block);

}

#endif


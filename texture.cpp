#include <fstream>
#include <iostream>

#include "texture.h"

Texture::Texture(const char* filename)
{
  std::ifstream inputFile;
  inputFile.open(filename, std::ifstream::in);

  if (inputFile.fail()) {
    std::cerr << "Unable to open file: " << filename << "\n";
    return;
  }


}
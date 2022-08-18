#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "model.h"

// https://dokipen.com/modern-opengl-part-11-creating-a-basic-obj-loader-and-loading-models-into-buffers/
Model::Model(const char* filename)
{
  std::ifstream inputFile;
  inputFile.open(filename, std::ifstream::in);

  if (inputFile.fail()) {
    std::cerr << "Unable to open file: " << filename << "\n";
    return;
  }

  std::string line;
  while (!inputFile.eof())
  {
    std::getline(inputFile, line);
    std::istringstream iss(line);
    char trash; int itrash;

    if (!line.compare(0, 2, "v "))
    {
      iss >> trash;
      Vec3f v;
      for (int i = 0; i < 3; ++i) iss >> v.raw[i];
      verts_.push_back(v);
    }
    else if (!line.compare(0, 2, "f "))
    {
      std::vector<int32_t> face;
      int vidx, vtidx;
      iss >> trash;
      // For this object we are only interested in the v/vt in v/vt/vn
      while (iss >> vidx >> trash >> vtidx >> trash >> itrash)
      {
        --vidx; // OBJ uses 1-based indexing
        --vtidx;
        face.push_back(vidx);
        face.push_back(vtidx);
      }
      faces_.push_back(face);
    }
    else if (!line.compare(0, 3, "vt "))
    {
      iss >> trash >> trash;
      Vec2f uv;
      for (int i = 0; i < 2; i++)
      {
        iss >> uv.raw[i];
      } 
      //std::cout << uv.x << ',' << uv.y << '\n';
      uvs_.push_back(uv);
    }
  }

  // TODO Remove unordered verts?
  for (int i = 0; i < faces_.size(); i++)
  {
    std::vector<int> face = faces_[i];
    for (int j = 0; j < 6; j+=2)
    {
      Vec3f v = verts_[face[j]];
      orderedVerts_.push_back(Vec3f(v.x, v.y, v.z));
      Vec2f uv = uvs_[face[j+1]];
      orderedUVs_.push_back(Vec2f(uv.x, uv.y));
    }
  }

  std::cout << "Number of verts:\t" << verts_.size() << '\n';
  std::cout << "Number of faces:\t" << faces_.size() << '\n';
}

Model::Model() // private ctor
{

}

Model Model::make_cube() 
{
  Model cube;
  
  // Add each face in turn to faces/verts and then copy as before
  
  float min_x, min_y, min_z;
  min_x = min_y = min_z = -0.5f;
  float max_x, max_y, max_z;
  max_x = max_y = max_z = 0.5f;
  
  std::vector<Vec3f> points = {
    { min_x, max_y, min_z },
    { max_x, max_y, min_z },
    { min_x, min_y, min_z },
    { max_x, min_y, min_z },
    { min_x, max_y, max_z },
    { max_x, max_y, max_z },
    { min_x, min_y, max_z },
    { max_x, min_y, max_z }
  };

  // TODO change unordered verts or assign verts ordered

  static uint16_t triangle_list[] = {
      0, 1, 2, // 0
      1, 3, 2,
      4, 6, 5, // 2
      5, 6, 7,
      0, 2, 4, // 4
      4, 2, 6,
      1, 5, 3, // 6
      5, 7, 3,
      0, 4, 1, // 8
      4, 5, 1,
      2, 3, 6, // 10
      6, 3, 7,
  };

  std::vector<int32_t> facev;
  for (int face = 0; face < 12; face++)
  {
    uint16_t vind = face * 3;
    cube.verts_.push_back(points[triangle_list[vind]]);   cube.orderedVerts_.push_back(points[triangle_list[vind]]);
    cube.verts_.push_back(points[triangle_list[vind+1]]); cube.orderedVerts_.push_back(points[triangle_list[vind+1]]);
    cube.verts_.push_back(points[triangle_list[vind+2]]); cube.orderedVerts_.push_back(points[triangle_list[vind+2]]);
    facev = { vind, vind + 1, vind + 2 };
    cube.faces_.push_back(facev);

    Vec2f uv((face / 2) / 6.0f, 0.5f);
    cube.uvs_.push_back(uv); cube.uvs_.push_back(uv); cube.uvs_.push_back(uv);
    cube.orderedUVs_.push_back(uv); cube.orderedUVs_.push_back(uv); cube.orderedUVs_.push_back(uv);

    //std::cout << "cubeu: " << uv.x << " cubev: " << uv.y << '\n';
  }

  std::cout << "Number of verts:\t" << cube.verts_.size() << '\n';
  std::cout << "Number of faces:\t" << cube.faces_.size() << '\n';

  return cube;
}

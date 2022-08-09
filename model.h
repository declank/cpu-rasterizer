#pragma once

#include <vector>
#include "math.h"

class Model
{
public:
  Model(const char* filename);
  ~Model() {}

  //size_t nverts() { return verts_.size(); }
  size_t nverts() { return orderedVerts_.size(); }
  size_t nfaces() { return faces_.size(); }

  Vec3f vert(int index) { return orderedVerts_[index]; }
  std::vector<int> face(int index) { return faces_[index]; }
  Vec2f uv(int index) { return orderedUVs_[index]; }

private:
  std::vector<Vec3f> verts_;
  std::vector<std::vector<int32_t> > faces_;
  std::vector<Vec2f> uvs_;

  std::vector<Vec3f> orderedVerts_;
  std::vector<Vec2f> orderedUVs_; // TODO position/uvs should be mixed together
};
#pragma once

#include <algorithm>
#include <cmath>

// forward declarations
template<typename T>
struct Vec4;

template<class T>
struct Mat4x1;

template<class T>
struct Mat4x4;

template<class T>
struct Mat4x4 {
  union {
    T raw[16];
    T raw2D[4][4];
  };

  // TODO Do we want to zero initialize when no values are given for our Matrices/Vectors
  Mat4x4() {} // TODO Does this even work?
  Mat4x4(T _val) //: raw({ (_val) }) {}
  {
    //raw = { _val };
    //memcpy(raw, &_val, 16);
    raw[0] = raw[1] = raw[2] = raw[3] = raw[4] = raw[5] = raw[6] = raw[7] =
      raw[8] = raw[9] = raw[10] = raw[11] = raw[12] = raw[13] = raw[14] = raw[15] =
      _val;
  }

  T* operator[](const int column)
  {
    return raw2D[column];
  }

  static Mat4x4 identity()
  {
    Mat4x4 result;
    for (int c = 0; c < 4; c++)
    {
      for (int r = 0; r < 4; r++)
      {
        result[c][r] = c == r ? 1.0f : 0.0f;
      }
    }

    return result;
  }

  Mat4x4<T> operator*(const Mat4x4<T>& rhs)
  {
    Mat4x4<T> result;
    for (int c = 0; c < 4; c++)
    {
      for (int r = 0; r < 4; r++)
      {
        result.raw2D[c][r] = 0.0f;
        for (int k = 0; k < 4; k++)
        {
          result.raw2D[c][r] += raw2D[k][r] * rhs.raw2D[c][k];
        }
      }
    }
    
    return result;
  }

  Mat4x1<T> operator*(const Mat4x1<T>& rhs)
  {
    Mat4x1<T> result;
    for (int r = 0; r < 4; r++)
    {
      result.raw[r] = raw2D[r][0] * rhs.raw[0]
        + raw2D[r][1] * rhs.raw[1]
        + raw2D[r][2] * rhs.raw[2]
        + raw2D[r][3] * rhs.raw[3];

    }

    return result;
  }

  //Mat4x4(T[]& _arr)

  /*inline Vec3<T> operator-(const Vec3<T>& rhs) const
  {
    return Vec3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
  }*/

  

  template <class > friend std::ostream& operator<<(std::ostream& s, Mat4x4<T>& v);
};

template<class T>
struct Mat4x1 {
  union {
    T raw[4];
    T raw2D[1][4];
  };

  Mat4x1() { Mat4x1(0); }
  Mat4x1(T _val)
  {
    raw[0] = raw[1] = raw[2] = raw[3] = _val;
  }

  // Mat4x1(Vec4<T>& rhs) { memcpy(raw, &_val, 4*sizeof(T)); }
  Mat4x1(const Vec4<T>& rhs) { std::copy_n(rhs.raw, 4, raw); }

  /*Mat4x1& operator=(Vec4<T>& rhs)
  {
    std::copy_n(rhs.raw, 4, raw);
  }*/

  template <class > friend std::ostream& operator<<(std::ostream& s, Mat4x1<T>& v);

  void printMatrix()
  {
    std::cout << "[ " << raw[0] << ", " << raw[1] << ", " << raw[2] << ", " << raw[3] << " ]\n";
  }

  //template <class > friend std::ostream& operator<<(std::ostream& s, Mat4x1<T>& v);
};

template<class T>
struct Mat1x4 {
  union {
    T raw[4];
    T raw2D[4][1];
  };

  Mat1x4() : raw({}) {}
  Mat1x4(T _val)
  {
    raw[0] = raw[1] = raw[2] = raw[3] = _val;
  }

  Mat1x4(const Vec4<T>& rhs) { std::copy_n(rhs.raw, 4, raw); }
  /*
  // Mat1x4(Vec4<T>& rhs) { memcpy(raw, &_val, 4*sizeof(T)); }
  // Mat1x4(Vec4<T>& rhs) { std::copy_n(rhs.raw, 4, raw); }

  Mat1x4& operator=(Vec4<T>& rhs)
  {
    std::copy_n(rhs.raw, 4, raw);
  }*/
};

template <class t> std::ostream& operator<<(std::ostream& s, Mat4x4<t>& v) {
  //s << "(" << v.x << ", " << v.y << ")\n";
  s << "[ " << v.raw[0] << ", " << v.raw[1] << ", " << v.raw[2] << ", " << v.raw[3] << ", \n"
    << "  " << v.raw[4] << ", " << v.raw[5] << ", " << v.raw[6] << ", " << v.raw[7] << ", \n"
    << "  " << v.raw[8] << ", " << v.raw[9] << ", " << v.raw[10] << ", " << v.raw[11] << ", \n"
    << "  " << v.raw[12] << ", " << v.raw[13] << ", " << v.raw[14] << ", " << v.raw[15] << " ]\n";
  return s;
}



/*template <class t> std::ostream& operator<<(std::ostream& s, Mat4x1<t>& v) {
  
}*/

typedef Mat4x4<int16_t> Mat4x4i16;
typedef Mat4x1<int16_t> Mat4x1i16;
typedef Mat1x4<int16_t> Mat1x4i16;
typedef Mat4x4<float> Mat4x4f;
typedef Mat4x1<float> Mat4x1f;

template<class T>
struct Vec4 {
  union {
    struct { T x, y, z, w; };
    T raw[4];
  };

  Vec4() : x(0), y(0) {}
  Vec4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
  Vec4(T _val) : x(_val), y(_val), z(_val), w(_val) {}

  inline Vec4<T> operator+(const Vec4<T>& rhs) const
  {
    return Vec4<T>(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w);
  }

  inline void operator+=(const Vec4<T>& rhs)
  {
    x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w;
  }

  /*inline Vec4<T> operator*(float f) const
  {
    return Vec4<T>(x * f, y * f, z * f);
  }*/

  inline Vec4<T> operator*(const Vec4<T>& rhs) const
  {
    return Vec4<T>(x * rhs.x, y * rhs.y, z * rhs.z, w * rhs.w);
  }

  inline Vec4<T> operator|(const Vec4<T>& rhs) const
  {
    return Vec4<T>(x | rhs.x, y | rhs.y, z | rhs.z, w | rhs.w);
  }
};

template<class T>
struct Vec3 {
  union {
    struct { T x, y, z; };
    T raw[3];
  };

  Vec3() : x(0), y(0) {}
  Vec3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

  inline Vec3<T> operator^(const Vec3<T>& v) const
  {
    return Vec3<T>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
  }

  inline Vec3<T> operator-(const Vec3<T>& rhs) const
  {
    return Vec3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
  }

  inline T operator*(const Vec3<T>& rhs) const
  {
    return x * rhs.x + y * rhs.y + z * rhs.z;
  }

  inline Vec3<T> operator*(float f) const
  {
    return Vec3<T>(x * f, y * f, z * f);
  }

  inline float norm() const {
    return std::sqrt(x * x + y * y + z * z);
  }

  Vec3<T>& normalize(T l = 1)
  {
    *this = (*this) * (l / norm());
    return *this;
  }

  inline T& operator[](const int i)
  {
    return raw[i];
  }

  template <class T> friend std::ostream& operator<<(std::ostream& s, Vec3<T>& v);


};

template<class T>
struct Vec2 {
  union {
    struct { T x, y; };
    T raw[2];
  };

  Vec2() : x(0), y(0) {}
  Vec2(T _x, T _y) : x(_x), y(_y) {}

  inline Vec2<T> operator+(const Vec2<T>& rhs) const
  {
    return Vec2<T>(x + rhs.x, y + rhs.y);
  }

  inline Vec2<T> operator-(const Vec2<T>& rhs) const
  {
    return Vec2<T>(x - rhs.x, y - rhs.y);
  }

  inline Vec2<T> operator*(float scalar) const
  {
    return Vec2<T>(x * scalar, y * scalar);
  }

  inline T operator[](const int i) const
  {
    return raw[i];
  }

  inline T& operator[](const int i)
  {
    return raw[i];
  }

  template <class > friend std::ostream& operator<<(std::ostream& s, Vec2<T>& v);
};

/*template <class t> struct Vec2 {
  union {
    struct { t u, v; };
    struct { t x, y; };
    t raw[2];
  };
  Vec2() : u(0), v(0) {}
  Vec2(t _u, t _v) : u(_u), v(_v) {}
  inline Vec2<t> operator +(const Vec2<t>& V) const { return Vec2<t>(u + V.u, v + V.v); }
  inline Vec2<t> operator -(const Vec2<t>& V) const { return Vec2<t>(u - V.u, v - V.v); }
  inline Vec2<t> operator *(float f)          const { return Vec2<t>(u * f, v * f); }
  //template <class > friend std::ostream& operator<<(std::ostream& s, Vec2<t>& v);
};*/

typedef Vec3<float> Vec3f;
typedef Vec2<int> Vec2i;
typedef Vec3<int> Vec3i;
typedef Vec2<float> Vec2f;
typedef Vec4<int16_t> Vec4i16;
typedef Vec2<int16_t> Vec2i16;
typedef Vec4<int> Vec4i;

template <typename T>
Vec3<T> cross(Vec3<T> v1, Vec3<T> v2) {
  return Vec3<T>(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

template <class t> std::ostream& operator<<(std::ostream& s, Vec2<t>& v) {
  s << "(" << v.x << ", " << v.y << ")\n";
  return s;
}

template <class t> std::ostream& operator<<(std::ostream& s, Vec3<t>& v) {
  s << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
  return s;
}


template <typename T>
T min3(T& a, T& b, T& c)
{
  T& minAB = (a > b) ? b : a;
  return (minAB > c ? c : minAB);
}

template <typename T>
T max3(T& a, T& b, T& c)
{
  T& minAB = (a > b) ? a : b;
  return (minAB > c ? minAB : c);
}

Mat4x1f v2m(const Vec3f& v);
Mat4x4f viewport(int x, int y, int width, int height);

template <class t> std::ostream& operator<<(std::ostream& s, Mat4x1<t>& v) {
  //s << "(" << v.x << ", " << v.y << ")\n";
  s << "[ " << v.raw[0] << ", " << v.raw[1] << ", " << v.raw[2] << ", " << v.raw[3] << " ]\n";
  return s;
}


#include <iostream>
#include "tests.h"

#ifdef RUN_TESTS

void runTests()
{
  std::cout << "Running tests\n";
  test_matrix_initialization();
  test_mpv_matrices();
  test_operator_bitshiftleft();
}

void test_matrix_initialization()
{
  Mat4x4f identity = Mat4x4f::identity();

  std::cout << identity;

  float expected[] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  };

  VERIFY(identity, expected);
}

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 800;

void test_mpv_matrices()
{
  //0.134781, -0.14723, 0.48805
  Vec3f v = { 0.134781, -0.14723, 0.48805 };
  
  Vec3f camera(0, 0, 3);

  Mat4x4f proj = Mat4x4f::identity();
  Mat4x4f vp = viewport(SCREEN_WIDTH / 8.0f, SCREEN_HEIGHT / 8.0f, SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.75);
  proj[3][2] = -1.0f / camera.z;

  std::cout << "proj:\n" << proj;
  std::cout << "vp:\n" << vp;


}

struct Test
{
  int x;
  int y;

  //template <class > friend std::ostream& operator<<(std::ostream& s, Mat4x4<T>& v);
  //
  friend std::ostream& operator<<(std::ostream& s, Test t);
};

std::ostream& operator<<(std::ostream& s, Test t)
{
  s << "x: " << t.x << " y: " << t.y << '\n';
  return s;
}


void test_operator_bitshiftleft()
{
  Test t1 = { 5, 10 };
  std::cout << "t1.x: " << t1.x << '\n';
  std::cout << "t1.y: " << t1.y << '\n';

  std::cout << "t1:\n" << t1;
}

#endif


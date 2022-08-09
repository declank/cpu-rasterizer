#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <iostream>
#include <map>
//#include <random>
#include <vector>

#include <SDL.h>

#include "math.h"
#include "model.h"
//#include "texture.h"
#include "tgaimage.h"
#include "timer.h"
#include "main.h"

//#define PRINT_PIXEL

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 800;
constexpr int POINTS = SCREEN_WIDTH * SCREEN_HEIGHT;

TimerContext timerContext;

int orient2d(Vec2i& a, Vec2i& b, Vec2i& c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

int orient2d(Vec3f& a, Vec3f& b, Vec2i& c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

int16_t orient2d(const Vec2i16& a, const Vec2i16& b, const Vec2i16& c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

int orient2d(Vec3i& a, Vec3i& b, Vec2i& c)
{
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

void line(std::vector<uint32_t>& pixels, int32_t x0, int32_t y0, int32_t x1, int32_t y1, const int32_t argb)
{
  // We need to transpose the image for steep lines so each y increment is correct
  bool steep = std::abs(x0 - x1) < std::abs(y0 - y1) ? true : false;
  if (steep)
  {
    std::swap(x0, y0);
    std::swap(x1, y1);
  }

  // If the x1 point is before x0 then we need to swap the points so it positively increments
  if (x1 < x0)
  {
    std::swap(x0, x1);
    std::swap(y0, y1);
  }

  int dx = x1 - x0;
  int dy = y1 - y0;
  int D = (2 * dy) - dx;
  int y = y0;
  int yinc = 1;

  if (dy < 0)
  {
    yinc = -1;
    dy = -dy;
  }

  for (int x = x0; x <= x1; ++x)
  {
    // if steep then transpose image back
    int32_t point = steep ? (x * SCREEN_WIDTH + y) : (y * SCREEN_WIDTH + x);

    if (point >= 0 && point < POINTS)
    {
      pixels[point] = argb;
    }

    if (D > 0)
    {
      y += yinc;
      D = D - (2 * dx);
    }
    D = D + (2 * dy);
  }
}

void horizontal_scanline_triangles(std::vector<uint32_t>& pixels, Vec2i t0, Vec2i t1, Vec2i t2, const int32_t argb)
{
  Timer t("horizontal_scanline_triangles - Horizontal Scanline Implementation", timerContext);

  // Draw from lower to upper
  if (t0.y > t1.y) std::swap(t0, t1);
  if (t0.y > t2.y) std::swap(t0, t2);
  if (t1.y > t2.y) std::swap(t1, t2);

  int totalHeight = t2.y - t0.y;
  int32_t point;
  for (int y = 0; y < totalHeight; y++)
  {
    bool second_half = y > t1.y - t0.y || t1.y == t0.y;
    int segmentHeight = second_half ? t2.y - t1.y : t1.y - t0.y;

    float alpha = (float)(y) / totalHeight;
    float beta = (float)(y - (second_half ? t1.y - t0.y : 0)) / segmentHeight;

    Vec2i A = t0 + (t2 - t0) * alpha;
    Vec2i B = second_half ? t1 + (t2 - t1) * beta : t0 + (t1 - t0) * beta;

    if (A.x > B.x) std::swap(A, B);
    for (int x = A.x; x <= B.x; x++)
    {
      point = (t0.y + y) * SCREEN_WIDTH + x;
      if (point >= 0 && point < POINTS)
      {
        pixels[point] = argb;
      }
    }
  }
}

void triangles(std::vector<uint32_t>& pixels, std::vector<int>& zbuffer, Vec3i t0, Vec3i t1, Vec3i t2, Vec2f* uvs, float intensity, TGAImage& texture) //const int32_t argb)
{
  Timer t("triangles - Inside-Outside Less Operations Per Pixel (Non-SIMD)", timerContext);

  int minX = min3(t0.x, t1.x, t2.x);
  int minY = min3(t0.y, t1.y, t2.y);
  int maxX = max3(t0.x, t1.x, t2.x);
  int maxY = max3(t0.y, t1.y, t2.y);
  //int maxX = minX; int maxY = minY;

  minX = minX < 0 ? 0 : minX;
  minY = minY < 0 ? 0 : minY;
  maxX = maxX > SCREEN_WIDTH ? SCREEN_WIDTH : maxX;
  maxY = maxY > SCREEN_HEIGHT ? SCREEN_HEIGHT : maxY;

  int A01 = t0.y - t1.y, B01 = t1.x - t0.x;
  int A12 = t1.y - t2.y, B12 = t2.x - t1.x;
  int A20 = t2.y - t0.y, B20 = t0.x - t2.x;

  // This calculation is used for our UV texture mapping
  float doubleArea = t0.x * A12 + t1.x * A20 + t2.x * (A01);

  Vec2i p(minX, minY);
  // Barycentric coordinations at minX/minY
  int w0_row = orient2d(t1, t2, p);
  int w1_row = orient2d(t2, t0, p);
  int w2_row = orient2d(t0, t1, p);

  // Flat shading with white color
  uint32_t argb = intensity * 255;
  argb = (argb << 16) | (argb << 8) | argb;

#ifdef PRINT_PIXEL
  std::cout << "min: " << minX << ',' << minY << '\n';
  std::cout << "max: " << maxX << ',' << maxY << '\n';

  std::cout << "uvs: " << uvs[0].x << ',' << uvs[0].y
    << '\t' << uvs[1].x << ',' << uvs[1].y
    << '\t' << uvs[2].x << ',' << uvs[2].y << '\n';
#endif

  // TODO Change union so access can be done using .u and .v instead of x/y
  float UINC = (uvs[0].x * (float)A12 + uvs[1].x * (float)A20 + uvs[2].x * (float)A01) / (float)doubleArea * (float)texture.width();
  float VINC = (uvs[0].y * (float)A12 + uvs[1].y * (float)A20 + uvs[2].y * (float)A01) / (float)doubleArea * (float)texture.width();

  float UINC_ROW = (uvs[0].x * (float)B12 + uvs[1].x * (float)B20 + uvs[2].x * (float)B01) / (float)doubleArea * (float)texture.width();
  float VINC_ROW = (uvs[0].y * (float)B12 + uvs[1].y * (float)B20 + uvs[2].y * (float)B01) / (float)doubleArea * (float)texture.width();

  float u_row = (uvs[0].x * (float)w0_row + uvs[1].x * (float)w1_row + uvs[2].x * (float)w2_row) / doubleArea * texture.width();
  float v_row = (uvs[0].y * (float)w0_row + uvs[1].y * (float)w1_row + uvs[2].y * (float)w2_row) / doubleArea * texture.width();

  int ZINC = t0.z * A12 + t1.z * A20 + t2.z * A01;
  int ZINC_ROW = t0.z * B12 + t1.z * B20 + t2.z * B01;

  int z_row = t0.z * w0_row + t1.z * w1_row + t2.z * w2_row;

  for (p.y = minY; p.y < maxY; p.y++)
  {
    // Barycentric at start of row
    int w0 = w0_row;
    int w1 = w1_row;
    int w2 = w2_row;

    float u = u_row;
    float v = v_row;
    float z = z_row;

    for (p.x = minX; p.x < maxX; p.x++)
    {
      if ((w0 | w1 | w2) >= 0)
      {
        //assert(p.x <= maxX);
#ifdef PRINT_PIXEL
        std::cout << p.x << ',' << p.y << ',' << w0 << ',' << w1 << ',' << w2 << '\n';
#endif
        //std::cout << "Non-SIMD Pixel:\t" << p.x << ", " << p.y << '\t' << w0 << ',' << w1 << ',' << w2 << '\n';
  //      int point = p.y * SCREEN_WIDTH + p.x;
  //      if (point >= 0 && point < POINTS)
  //        pixels[point] = argb;
        int point = p.y * SCREEN_WIDTH + p.x;
        // TODO factor this out
        //int zpoint = t0.z * w0 + t1.z * w1 + t2.z * w2;
        if (zbuffer[point] < z)
        {
          // TODO calculate the values instead at P and increment similar to w0/w0_row etc.
          //int16_t u = (uvs[0].x * (float)w0 + uvs[1].x * (float)w1 + uvs[2].x * (float)w2) / doubleArea * texture.width();
          //int16_t v = (1.0f - (uvs[0].y * (float)w0 + uvs[1].y * (float)w1 + uvs[2].y * (float)w2) / doubleArea) * texture.height();
#ifdef PRINT_PIXEL
          std::cout << "uv: " << u << ',' << v << '\n';
#endif

          //uint32_t argb = (t0R * w0 + t1R * w1 + t2R * w2) << 16 + (t0G * w0 + t1G * w1 + t2G * w2) << 8 + (t0B * w0 + t1B * w1 + t2B * w2);
          /*std::cout << "uvs[0]: " << uvs[0].x << ',' << uvs[0].y << '\n';
          std::cout << "uvs[1]: " << uvs[1].x << ',' << uvs[1].y << '\n';
          std::cout << "uvs[2]: " << uvs[2].x << ',' << uvs[2].y << '\n';
          std::cout << "U:      " << (uvs[0].x * w0 + uvs[1].x * w1 + uvs[2].x * w2) / 2.f << '\n';
          std::cout << "V:      " << (uvs[0].y * w0 + uvs[1].y * w1 + uvs[2].y * w2) / 2.f << '\n';*/


          //float u = (uvs[0].x * w0 + uvs[1].x * w1 + uvs[2].x * w2) / 2.f;
          //float v = (uvs[0].y * w0 + uvs[1].y * w1 + uvs[2].y * w2) / 2.f;

          //TGAColor color = texture.get(u * texture.width(), v * texture.height());
          //uint32_t argb = 0xff;
          //uint32_t argb = (color.bgra[2] << 16) | (color.bgra[1] << 8) | color.bgra[0];
          /*std::cout << "uv:   " << u * texture.width() << ',' << v * texture.height() << '\n';
          std::cout << "argb: " << std::bitset<24>(argb) << '\n';*/

          TGAColor color = texture.get(u, texture.height() - v);
          argb = ((int)(color.bgra[2] * intensity) << 16) | ((int)(color.bgra[1] * intensity) << 8) | (int)(color.bgra[0] * intensity);


          pixels[point] = argb;
          zbuffer[point] = z;
        }
      }

      w0 += A12;
      w1 += A20;
      w2 += A01;

      u += UINC;
      v += VINC;
      z += ZINC;
    }

    w0_row += B12;
    w1_row += B20;
    w2_row += B01;

    u_row += UINC_ROW;
    v_row += VINC_ROW;
    z_row += ZINC_ROW;
  }

  //__debugbreak();
}

int totalYDiffs = 0;
int totalXDiffs = 0;

void triangles_simd(std::vector<uint32_t>& pixels, Vec2i t0, Vec2i t1, Vec2i t2, const int32_t argb)
{
  Timer t("triangles - Inside-Outside (SIMD)", timerContext);


  int minX = min3(t0.x, t1.x, t2.x);
  int minY = min3(t0.y, t1.y, t2.y);
  int maxX = max3(t0.x, t1.x, t2.x);
  int maxY = max3(t0.y, t1.y, t2.y);

  minX = minX < 0 ? 0 : minX;
  minY = minY < 0 ? 0 : minY;
  maxX = maxX > SCREEN_WIDTH ? SCREEN_WIDTH : maxX;
  maxY = maxY > SCREEN_HEIGHT ? SCREEN_HEIGHT : maxY;

  //int maxX = minX; int maxY = minY;

  int A01 = t0.y - t1.y, B01 = t1.x - t0.x;
  int A12 = t1.y - t2.y, B12 = t2.x - t1.x;
  int A20 = t2.y - t0.y, B20 = t0.x - t2.x;

  Vec2i p(minX, minY);
  // Barycentric coordinations at minX/minY
  int16_t w0_row = orient2d(t1, t2, p);
  int16_t w1_row = orient2d(t2, t0, p);
  int16_t w2_row = orient2d(t0, t1, p);



  __m256i multiplicand = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  __m256i vA12 = _mm256_set1_epi16(A12); vA12 = _mm256_mullo_epi16(vA12, multiplicand);
  __m256i vA20 = _mm256_set1_epi16(A20); vA20 = _mm256_mullo_epi16(vA20, multiplicand);
  __m256i vA01 = _mm256_set1_epi16(A01); vA01 = _mm256_mullo_epi16(vA01, multiplicand);

  __m256i multiplicandStepX = _mm256_setr_epi16(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  __m256i w0StepX = _mm256_set1_epi16(A12); w0StepX = _mm256_mullo_epi16(w0StepX, multiplicandStepX);
  __m256i w1StepX = _mm256_set1_epi16(A20); w1StepX = _mm256_mullo_epi16(w1StepX, multiplicandStepX);
  __m256i w2StepX = _mm256_set1_epi16(A01); w2StepX = _mm256_mullo_epi16(w2StepX, multiplicandStepX);

  for (p.y = minY; p.y < maxY; p.y++)
  {
    // Barycentric at start of row
    /*int16_t w0 = w0_row;
    int16_t w1 = w1_row;
    int16_t w2 = w2_row;*/

    /*__m256i w0 = _mm256_set1_epi16(w0_row); w0 = _mm256_mullo_epi16(w0, multiplicand);
    __m256i w1 = _mm256_set1_epi16(w1_row); w0 = _mm256_mullo_epi16(w1, multiplicand);
    __m256i w2 = _mm256_set1_epi16(w2_row); w0 = _mm256_mullo_epi16(w2, multiplicand);
    */

    __m256i w0 = _mm256_set1_epi16(w0_row); w0 = _mm256_add_epi16(w0, vA12);
    __m256i w1 = _mm256_set1_epi16(w1_row); w1 = _mm256_add_epi16(w1, vA20);
    __m256i w2 = _mm256_set1_epi16(w2_row); w2 = _mm256_add_epi16(w2, vA01);

    for (p.x = minX; p.x <= maxX; p.x += 16)
    {
      /*if ((w0 | w1 | w2) >= 0)
      {
        pixels[p.y * SCREEN_WIDTH + p.x] = argb;
      }*/

      short arrayw0[16], arrayw1[16], arrayw2[16];
      _mm256_storeu_epi16(arrayw0, w0);
      _mm256_storeu_epi16(arrayw1, w1);
      _mm256_storeu_epi16(arrayw2, w2);

      for (int i = 0; i < 16; i++)
      {
        if ((arrayw0[i] | arrayw1[i] | arrayw2[i]) >= 0)
        {
          //assert((p.x + i) <= maxX);
#ifdef PRINT_PIXEL
          std::cout << p.x + i << ',' << p.y << ',' << arrayw0[i] << ',' << arrayw1[i] << ',' << arrayw2[i] << '\n';
#endif
          //if ((p.x + i) > maxX) __debugbreak();
          if ((p.x + i) > maxX) break;
          pixels[p.y * SCREEN_WIDTH + p.x + i] = argb;
          //std::cout << "SIMD Pixel:  \t" << p.x + i << ", " << p.y << '\t' << arrayw0[i] << ',' << arrayw1[i] << ',' << arrayw2[i] << '\n';
        }
      }

      w0 = _mm256_add_epi16(w0, w0StepX);
      w1 = _mm256_add_epi16(w1, w1StepX);
      w2 = _mm256_add_epi16(w2, w2StepX);
    }

    w0_row += B12;
    w1_row += B20;
    w2_row += B01;

  }
}

struct Edge {
  static const int stepXSize = 4;
  static const int stepYSize = 1;

  Vec4i oneStepX;
  Vec4i oneStepY;

  Vec4i init(const Vec2i& v0, const Vec2i& v1, const Vec2i& v2);
};

Vec4i Edge::init(const Vec2i& v0, const Vec2i& v1, const Vec2i& origin)
{
  int A = v0.y - v1.y, B = v1.x - v0.x;
  int C = v0.x * v1.y - v0.y * v1.x;

  // Step deltas
  oneStepX = Vec4i(A * stepXSize);
  oneStepY = Vec4i(B * stepYSize);

  // x/y values for initial pixel block
  Vec4i x = Vec4i(origin.x) + Vec4i(0, 1, 2, 3);
  Vec4i y = Vec4i(origin.y);

  // Edge function values at origin
  return Vec4i(A) * x + Vec4i(B) * y + Vec4i(C);
}

bool anyGtZero(const Vec4i& mask)
{
  for (int i = 0; i < Edge::stepXSize; i++)
  {
    if (mask.raw[i] >= 0) return true;
  }

  return false;
}

void renderPixels(const Vec2i& p, /*const Vec4i& w0, const Vec4i& w1, const Vec4i& w2,*/ const Vec4i& mask, std::vector<uint32_t>& pixels, const int32_t argb)
{
  for (int i = 0; i < Edge::stepXSize; i++)
  {
    if (mask.raw[i] >= 0)
    {
      pixels[p.y * SCREEN_WIDTH + p.x] = argb;
    }
  }
}

Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P) {
  Vec3f s[2];
  for (int i = 2; i--; ) {
    s[i][0] = C[i] - A[i];
    s[i][1] = B[i] - A[i];
    s[i][2] = A[i] - P[i];
  }
  Vec3f u = cross(s[0], s[1]);
  if (std::abs(u[2]) > 1e-2) // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
  return Vec3f(-1, 1, 1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

void triangles_tinyrenderer(std::vector<uint32_t>& pixels, Vec3f* pts, const int32_t argb, float* zbuffer)
{
  Timer t("triangles - Inside-Outside Based On TinyRenderer", timerContext);

  Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
  Vec2f clamp(SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      bboxmin[j] = std::max(0.f, std::min(bboxmin[j], pts[i][j]));
      bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts[i][j]));
    }
  }
  Vec3f P;
  for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
    for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
      Vec3f bc_screen = barycentric(pts[0], pts[1], pts[2], P);
      if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
      P.z = 0;
      for (int i = 0; i < 3; i++) P.z += pts[i][2] * bc_screen[i];
      if (zbuffer[int(P.x + P.y * SCREEN_WIDTH)] < P.z) {
        zbuffer[int(P.x + P.y * SCREEN_WIDTH)] = P.z;
        pixels[int(P.x + P.y * SCREEN_WIDTH)] = 0xff;
        //image.set(P.x, P.y, color);
      }
    }
  }
}

int maintest(int argc, char* argv[])
{
  //Model model("objs/african_head.obj");

  std::vector<uint32_t> pixels(SCREEN_WIDTH * SCREEN_HEIGHT, 0);
  int32_t icolor = 0xFF;

  Vec2i scrint[3] = { Vec2i(800, 700), Vec2i(810, 710), Vec2i(820, 740) };
  std::vector<float> zbuffer(std::numeric_limits<float>::min());

  std::cout << "================ NON SIMD ===================\n";
  //triangles(pixels, zbuffer, scrint[0], scrint[1], scrint[2], icolor << 16 | icolor << 8 | icolor);
  std::cout << "================== SIMD =====================\n";
  triangles_simd(pixels, scrint[0], scrint[1], scrint[2], icolor << 16 | icolor << 8 | icolor);

  return 0;

}

Vec3f m2v(const Mat4x4f& m)
{
  float invW = 1.0f / m.raw2D[3][0];
  return Vec3f(m.raw2D[0][0] * invW, m.raw2D[1][0] * invW, m.raw2D[2][0] * invW);
}

Mat4x1f v2m(const Vec3f& v)
{
  Mat4x1f m;
  m.raw2D[0][0] = v.x;
  m.raw2D[1][0] = v.y;
  m.raw2D[2][0] = v.z;
  m.raw2D[3][0] = 1.0f;
  return m;
}

int main(int argc, char* argv[])
{
  SDL_Init(SDL_INIT_VIDEO);

  SDL_Window* window = SDL_CreateWindow("Software Rasterizer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_RENDERER_ACCELERATED);

  if (window == nullptr)
  {
    std::cerr << "Error: Could not create window: " << SDL_GetError() << std::endl;
    return 1;
  }

  SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
  SDL_Texture* output = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);

  SDL_Event event;
  bool running = true;

  std::vector<uint32_t> pixels(SCREEN_WIDTH * SCREEN_HEIGHT, 0);
  int color = 0;

  Model model("objs/african_head.obj");
  //Model model("objs/mariohead.obj");
  //Model model("objs/CTcras.obj");
  //Model model("objs/stanford-bunny.obj");
  //Model model("objs/helmet.obj");

  //Texture texture("tgas/african_head_diffuse.tga");
  TGAImage texture;
  texture.read_tga_file("tgas/african_head_diffuse.tga");

  std::vector<int> zbuffer(SCREEN_WIDTH * SCREEN_HEIGHT);// , std::numeric_limits<int>::min());
  std::vector<float> fzbuffer(SCREEN_WIDTH * SCREEN_HEIGHT);//, -std::numeric_limits<float>::min());

  Vec3f camera(0, 0, 3);

  int numberOfRuns = 0;

  while (running)
  {
    if (numberOfRuns++ == 100)
    {
      running = false;
      break;
    }

    while (SDL_PollEvent(&event))
    {
      switch (event.type) {
      case SDL_QUIT:
        running = false;
        break;
      }
    }

    // Clear pixels
    for (auto& pixel : pixels) pixel = 0;
    for (auto& z : zbuffer) z = std::numeric_limits<int>::min();

    //constexpr float widthOffset = SCREEN_WIDTH / 2.0f;
    //constexpr float heightOffset = SCREEN_HEIGHT / 2.0f;

    constexpr float widthOffset = SCREEN_WIDTH / 2.0f;
    constexpr float heightOffset = SCREEN_HEIGHT / 2.0f;

    // Wireframe drawing

    {
      for (int i = 0; i < model.nfaces(); ++i)
      {
        auto face = model.face(i);
        for (int j = 0; j < 3; ++j)
        {
          //Vec3f v0 = model.vert(face[j]);
          //Vec3f v1 = model.vert(face[(j + 1) % 3]);

          Vec3f v0 = model.vert(i * 3 + j);
          Vec3f v1 = model.vert(i * 3 + ((j + 1) % 3));

          int32_t x0 = (int32_t)((v0.x + 1.0f) * widthOffset);
          int32_t y0 = (int32_t)((v0.y + 1.0f) * heightOffset);
          int32_t x1 = (int32_t)((v1.x + 1.0f) * widthOffset);
          int32_t y1 = (int32_t)((v1.y + 1.0f) * heightOffset);

          //line(pixels, x0, y0, x1, y1, 0xffffff);

        }
      }
    }

    // Flat shaded polygons with z-buffer
    {
      Timer t("Overall - Face Drawing", timerContext);

      Vec3f light_dir(0, 0, -1);

      size_t numberOfFaces = model.nfaces();
      for (int i = 0; i < numberOfFaces; i++)
      {
        Vec3f scr[3];
        Vec2i scrint[3];
        Vec3f world[3];
        Vec3i scr3i[3];
        Vec2f uvs[3];
        for (int j = 0; j < 3; j++)
        {
          Vec3f v = model.vert(i * 3 + j);
          //scrint[j] = Vec2i((v.x + 1.0f) * widthOffset, (v.y + 1.0f) * heightOffset);
          scr[j] = Vec3f((v.x + 1.0f) * widthOffset, (v.y + 1.0f) * heightOffset, v.z);
          scr3i[j] = Vec3i((v.x + 1.0f) * widthOffset, (v.y + 1.0f) * heightOffset, v.z);
          //scr3i[j] = Vec3i((v.x + 4.0f) * widthOffset * 0.25f, (v.y + 4.0f) * heightOffset * 0.25f, (v.z + 4.0f) * 0.25f);
          world[j] = v;
          uvs[j] = model.uv(i * 3 + j);
        }

        Vec3f nor = (world[2] - world[0]) ^ (world[1] - world[0]);
        nor.normalize();

        float intensity = nor * light_dir;
        if (intensity > 0) // is the intensity check back face culling?
        {
          //int tX = texture.width() * uvs[0].x;
          //int tY = texture.height() * uvs[0].y;
          //TGAColor color = texture.get(tX, tY);
          int icolor = intensity * 255;
          //int icolor = (color.bgra[2] << 16) | (color.bgra[1] << 8) | (color.bgra[0]);
          //triangles(pixels, zbuffer, scrint[0], scrint[1], scrint[2], icolor << 16 | icolor << 8 | icolor);
          triangles(pixels, zbuffer, scr3i[0], scr3i[1], scr3i[2], uvs, intensity, texture);
          //triangles_tinyrenderer(pixels, scr, icolor << 16 | icolor << 8 | icolor, fzbuffer.data());

          //triangles_simd(pixels, scrint[0], scrint[1], scrint[2], icolor << 16 | icolor << 8 | icolor);
          //triangles_tinyrenderer(pixels, scrint, icolor << 16 | icolor << 8 | icolor);

        }
      }
    }

    SDL_UpdateTexture(output, nullptr, pixels.data(), 4 * SCREEN_WIDTH);
    SDL_RenderCopyEx(renderer, output, nullptr, nullptr, 0, nullptr, SDL_FLIP_VERTICAL);
    SDL_RenderPresent(renderer);
    //SDL_Delay(15000);
    //SDL_Delay(1000 / 60);
  }

  std::cout << "Average X Diff: " << totalXDiffs / model.nfaces() << '\n';
  std::cout << "Average Y Diff: " << totalYDiffs / model.nfaces() << '\n';

  timerContext.printTimings();

  SDL_DestroyWindow(window);
  SDL_Quit();

  std::cin.get();

  return 0;
}

int main3(int argc, char* argv[])
{
  std::vector<uint32_t>pixels; Vec2i t0; Vec2i t1; Vec2i t2; const int32_t argb = 0xff;

  t0 = { 20, 10 };
  t1 = { 40, 30 };
  t2 = { 30, 20 };

  int minX = min3(t0.x, t1.x, t2.x);
  int minY = min3(t0.y, t1.y, t2.y);
  int maxX = max3(t0.x, t1.x, t2.x);
  int maxY = max3(t0.y, t1.y, t2.y);

  int A01 = t0.y - t1.y, B01 = t1.x - t0.x;
  int A12 = t1.y - t2.y, B12 = t2.x - t1.x;
  int A20 = t2.y - t0.y, B20 = t0.x - t2.x;

  Vec2i p(minX, minY);
  // Barycentric coordinations at minX/minY
  int w0_row = orient2d(t1, t2, p);
  int w1_row = orient2d(t2, t0, p);
  int w2_row = orient2d(t0, t1, p);

  for (p.y = minY; p.y < maxY; p.y++)
  {
    //std::cout << "p.y = " << p.y << '\n';
    // Barycentric at start of row
    int w0 = w0_row;
    int w1 = w1_row;
    int w2 = w2_row;

    for (p.x = minX; p.x <= maxX; p.x++)
    {
      //std::cout << "p.x = " << p.x;
      if ((w0 | w1 | w2) >= 0)
        //if (w0 >= 0 && w1 >= 0 && w2 >= 0)
      {
        std::cout << "" << p.x << "," << p.y << '\n';
        //pixels[p.y * SCREEN_WIDTH + p.x] = argb;
      }

      //std::cout << '\n';

      w0 += A12;
      w1 += A20;
      w2 += A01;
    }

    w0_row += B12;
    w1_row += B20;
    w2_row += B01;
  }

  return 0;
}
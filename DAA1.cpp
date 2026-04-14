// ============================================================
//  FIREFLY ALGORITHM VISUALIZER — MEGA UPGRADE
//  Raylib 5.x | C++17
//  Build: g++ firefly_visualizer.cpp -o fa_viz -lraylib -lGL -lm -lpthread -ldl
//  -lrt -lX11 Or on Windows with raylib installed: g++ firefly_visualizer.cpp
//  -o fa_viz.exe -lraylib -lopengl32 -lgdi32 -lwinmm
// ============================================================

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ============================================================
// SECTION 1: WINDOW / LAYOUT CONSTANTS
// ============================================================
static const int WIN_W = 1600;
static const int WIN_H = 900;
static const int N = 8;
static const int POP = 24;
static const int MAX_ITER = 120;
static const int MAX_FIT = N * (N - 1) / 2; // 28

// Layout regions
static const Rectangle BOARD_RECT = {20, 60, 480, 480};
static const Rectangle SWARM_RECT = {520, 60, 540, 300};
static const Rectangle GRAPH_RECT = {520, 375, 540, 250};
static const Rectangle METRICS_RECT = {1075, 60, 500, 300};
static const Rectangle HEATMAP_RECT = {1075, 375, 500, 250};
static const Rectangle CTRL_RECT = {20, 785, WIN_W - 40, 80};

// ============================================================
// SECTION 2: SHADERS
// ============================================================
static const char *BLOOM_FRAG = R"(
#version 330
in vec2 fragTexCoord;
uniform sampler2D texture0;
uniform vec2 resolution;
uniform float intensity;
out vec4 finalColor;
void main(){
    vec2 uv = fragTexCoord;
    vec4 c = texture(texture0, uv);
    float wt[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec4 res = c * wt[0];
    vec2 off = vec2(1.3846153846, 1.3846153846) / resolution;
    for(int i=1;i<5;i++){
        res += texture(texture0, uv + off*float(i)) * wt[i];
        res += texture(texture0, uv - off*float(i)) * wt[i];
    }
    finalColor = c + res * intensity;
}
)";

static const char *VIGNETTE_FRAG = R"(
#version 330
in vec2 fragTexCoord;
uniform sampler2D texture0;
out vec4 finalColor;
void main(){
    vec4 c = texture(texture0, fragTexCoord);
    vec2 uv = fragTexCoord - 0.5;
    float v = smoothstep(0.85, 0.35, length(uv) * 1.4);
    finalColor = vec4(c.rgb * v, c.a);
}
)";

static const char *GLOW_FRAG = R"(
#version 330
in vec2 fragTexCoord;
uniform sampler2D texture0;
uniform vec2 center;
uniform float radius;
uniform vec4 glowColor;
out vec4 finalColor;
void main(){
    vec4 c = texture(texture0, fragTexCoord);
    float d = distance(fragTexCoord, center);
    float g = exp(-d * d / (radius * radius * 0.01)) * 0.6;
    finalColor = c + glowColor * g;
}
)";

// ============================================================
// SECTION 3: COLOUR PALETTE
// ============================================================
struct Palette {
  Color bg = {8, 10, 20, 255};
  Color panel = {16, 18, 32, 255};
  Color panelBord = {35, 40, 65, 255};
  Color text = {200, 210, 230, 255};
  Color dimText = {100, 110, 140, 255};
  Color axis = {60, 65, 90, 255};

  Color lightSq = {232, 237, 209, 255};
  Color darkSq = {90, 130, 60, 255};

  Color orig = {60, 140, 255, 255}; // electric blue
  Color origDim = {30, 70, 130, 255};
  Color mod = {255, 130, 30, 255}; // amber
  Color modDim = {130, 65, 15, 255};

  Color gold = {255, 220, 40, 255};
  Color goldGlow = {255, 190, 0, 100};
  Color conflict = {255, 50, 60, 255};
  Color safe = {60, 220, 90, 255};
  Color accent = {180, 80, 255, 255};

  Color heatLow = {15, 20, 50, 255};
  Color heatMid = {100, 40, 200, 255};
  Color heatHigh = {255, 180, 40, 255};
};

static Palette PAL;

// ============================================================
// SECTION 4: UTILITY
// ============================================================
// Redundant ColorAlpha removed (provided by raylib.h)
static Color LerpColor(Color a, Color b, float t) {
  return {(unsigned char)Lerp(a.r, b.r, t), (unsigned char)Lerp(a.g, b.g, t),
          (unsigned char)Lerp(a.b, b.b, t), (unsigned char)Lerp(a.a, b.a, t)};
}
static Color FitnessColor(float f) { // 0..28 → red to gold
  float t = f / MAX_FIT;
  if (t < 0.5f)
    return LerpColor(PAL.conflict, PAL.mod, t * 2.f);
  return LerpColor(PAL.mod, PAL.gold, (t - 0.5f) * 2.f);
}
static void DrawRoundedBorder(Rectangle r, float round, int segs, float thick,
                              Color c) {
  DrawRectangleRoundedLines(r, round, segs, thick, c);
}

// Smooth noise for ambient shimmer
static float hash21(float x, float y) {
  float h = x * 127.1f + y * 311.7f;
  return fmodf(sinf(h) * 43758.5453f, 1.f);
}

// ============================================================
// SECTION 5: ALGORITHM ENGINE
// ============================================================
static int calcFitness(const std::vector<int> &pos) {
  int atk = 0;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      if (std::abs(pos[i] - pos[j]) == std::abs(i - j))
        ++atk;
  return MAX_FIT - atk;
}

static void repairPerm(std::vector<int> &pos, std::mt19937 &rng) {
  std::vector<int> cnt(N, 0);
  for (int v : pos)
    if (v >= 0 && v < N)
      cnt[v]++;
  std::vector<int> miss;
  for (int v = 0; v < N; v++)
    if (cnt[v] == 0)
      miss.push_back(v);
  std::shuffle(miss.begin(), miss.end(), rng);
  int mi = 0;
  for (int i = 0; i < N; i++) {
    if (pos[i] < 0 || pos[i] >= N) {
      pos[i] = miss[mi++];
      continue;
    }
    if (cnt[pos[i]] > 1) {
      cnt[pos[i]]--;
      pos[i] = miss[mi++];
    }
  }
}

static std::vector<int> heuristicInit(std::mt19937 &rng) {
  std::vector<int> pos(N, -1);
  std::vector<bool> used(N, false);
  for (int row = 0; row < N; row++) {
    std::vector<int> cols;
    for (int c = 0; c < N; c++)
      if (!used[c])
        cols.push_back(c);
    std::shuffle(cols.begin(), cols.end(), rng);
    int bc = 999, bCol = cols[0];
    for (int c : cols) {
      int conf = 0;
      for (int r2 = 0; r2 < row; r2++)
        if (std::abs(pos[r2] - c) == std::abs(r2 - row))
          conf++;
      if (conf < bc) {
        bc = conf;
        bCol = c;
      }
    }
    pos[row] = bCol;
    used[bCol] = true;
  }
  return pos;
}

struct Trail {
  std::deque<Vector2> pts;
  static const int MAX = 20;
  void push(Vector2 p) {
    pts.push_back(p);
    if ((int)pts.size() > MAX)
      pts.pop_front();
  }
  void clear() { pts.clear(); }
};

struct Firefly {
  std::vector<int> position;
  float fitness = 0.f;
  Vector2 screenPos = {0, 0};
  Vector2 targetPos = {0, 0};
  Trail trail;
  float pulsePhase = 0.f;
  float wobble = 0.f;
  bool isElite = false;
  Color tint = WHITE;
};

struct RunStats {
  std::vector<float> bestPerIter;
  std::vector<float> avgPerIter;
  std::vector<float> worstPerIter;
  int iterToOptimal = -1;
  int optimalCount = 0;
  float bestFit = 0.f;
  float avgFit = 0.f;
  int totalIter = 0;
  // Heat map: how often each cell [row][col] is occupied by any queen
  std::array<std::array<int, N>, N> heatMap = {};
};

struct FAEngine {
  std::vector<Firefly> pop;
  RunStats stats;
  int iter = 0;
  bool done = false;
  bool inited = false;
  float alpha0 = 0.5f, alpha = 0.5f;
  float beta0 = 1.0f, gamma0 = 0.1f;
  bool adaptive = false, elitism = false, heurInit = false;
  std::mt19937 rng;
  bool isModified = false;
  Color col = WHITE;

  void init(bool modified, Color c) {
    std::random_device rd;
    rng.seed(rd());
    iter = 0;
    done = false;
    inited = true;
    isModified = modified;
    col = c;
    stats = RunStats{};
    adaptive = elitism = heurInit = modified;
    alpha0 = modified ? 0.9f : 0.5f;
    alpha = alpha0;
    std::vector<int> base(N);
    std::iota(base.begin(), base.end(), 0);
    pop.resize(POP);
    for (auto &ff : pop) {
      ff.position = heurInit ? heuristicInit(rng) : base;
      if (!heurInit)
        std::shuffle(ff.position.begin(), ff.position.end(), rng);
      ff.fitness = (float)calcFitness(ff.position);
      ff.trail.clear();
      ff.pulsePhase = (float)GetRandomValue(0, 628) / 100.f;
      ff.wobble = (float)GetRandomValue(0, 628) / 100.f;
      ff.tint = col;
    }
    layoutSwarm();
    for (auto &ff : pop)
      ff.screenPos = ff.targetPos;
  }

  void layoutSwarm() {
    const float sx = SWARM_RECT.x + 14.f;
    const float sy = SWARM_RECT.y + 40.f;
    const float pw = SWARM_RECT.width - 28.f;
    const float ph = SWARM_RECT.height - 50.f;
    const int cols = 6;
    const int rows = (POP + cols - 1) / cols;
    for (int i = 0; i < (int)pop.size(); i++) {
      int row = i / cols, c = i % cols;
      float fx = sx + (c + 0.5f) * (pw / cols);
      float fy = sy + (row + 0.5f) * (ph / rows);
      pop[i].targetPos = {fx, fy};
    }
  }

  void step() {
    if (!inited || done)
      return;
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    // Mark elites
    if (elitism) {
      auto sorted = pop;
      std::sort(sorted.begin(), sorted.end(),
                [](auto &a, auto &b) { return a.fitness > b.fitness; });
      for (auto &ff : pop)
        ff.isElite = false;
      for (int e = 0; e < 2 && e < (int)sorted.size(); e++) {
        auto &best = sorted[e];
        for (auto &ff : pop)
          if (ff.position == best.position) {
            ff.isElite = true;
            break;
          }
      }
    }
    std::vector<Firefly> elites;
    if (elitism) {
      for (auto &ff : pop)
        if (ff.isElite)
          elites.push_back(ff);
    }
    if (adaptive)
      alpha = alpha0 * (1.f - (float)iter / MAX_ITER) +
              0.05f * ((float)iter / MAX_ITER);

    for (int i = 0; i < POP; i++) {
      for (int j = 0; j < POP; j++) {
        if (pop[j].fitness > pop[i].fitness) {
          float r2 = 0.f;
          for (int k = 0; k < N; k++) {
            float d = (float)(pop[i].position[k] - pop[j].position[k]);
            r2 += d * d;
          }
          float beta = beta0 * expf(-gamma0 * r2 / (N * N));
          std::vector<int> np(N);
          for (int k = 0; k < N; k++) {
            float v = pop[i].position[k] +
                      beta * (pop[j].position[k] - pop[i].position[k]) +
                      alpha * (u01(rng) - 0.5f) * N;
            np[k] = std::max(0, std::min(N - 1, (int)roundf(v)));
          }
          repairPerm(np, rng);
          float nf = (float)calcFitness(np);
          if (nf >= pop[i].fitness) {
            pop[i].position = np;
            pop[i].fitness = nf;
          }
        }
      }
    }
    if (elitism && !elites.empty()) {
      std::sort(pop.begin(), pop.end(),
                [](auto &a, auto &b) { return a.fitness < b.fitness; });
      for (int e = 0; e < (int)elites.size(); e++)
        pop[e] = elites[e];
    }

    float best = 0.f, worst = 99.f, avg = 0.f;
    for (auto &ff : pop) {
      if (ff.fitness > best)
        best = ff.fitness;
      if (ff.fitness < worst)
        worst = ff.fitness;
      avg += ff.fitness;
      if (ff.fitness >= (float)MAX_FIT)
        stats.optimalCount++;
      // Heat map
      for (int r = 0; r < N; r++) {
        if (ff.position[r] >= 0 && ff.position[r] < N)
          stats.heatMap[r][ff.position[r]]++;
      }
    }
    avg /= POP;
    stats.bestFit = std::max(stats.bestFit, best);
    stats.avgFit = avg;
    stats.bestPerIter.push_back(best);
    stats.avgPerIter.push_back(avg);
    stats.worstPerIter.push_back(worst);
    if (stats.iterToOptimal < 0 && best >= (float)MAX_FIT)
      stats.iterToOptimal = iter;
    layoutSwarm();
    ++iter;
    ++stats.totalIter;
    if (iter >= MAX_ITER)
      done = true;
  }

  Firefly getBest() const {
    return *std::max_element(pop.begin(), pop.end(), [](auto &a, auto &b) {
      return a.fitness < b.fitness;
    });
  }
};

// ============================================================
// SECTION 6: PARTICLE SYSTEM
// ============================================================
struct Particle {
  Vector2 pos, vel;
  Color col;
  float life, maxLife, size;
  int type; // 0=confetti, 1=spark, 2=ember
};

struct ParticleSystem {
  std::vector<Particle> particles;
  void emit(Vector2 origin, int count, int type, Color base) {
    for (int i = 0; i < count; i++) {
      float angle = (float)GetRandomValue(0, 628) / 100.f;
      float spd = (float)GetRandomValue(60, 400);
      Particle p;
      p.pos = origin;
      p.vel = {cosf(angle) * spd,
               sinf(angle) * spd - (float)GetRandomValue(100, 300)};
      p.col = ColorFromHSV((float)GetRandomValue(0, 360), 0.8f, 0.95f);
      if (type == 1)
        p.col = LerpColor(base, WHITE, 0.6f);
      p.life = (float)GetRandomValue(80, 180) / 100.f;
      p.maxLife = p.life;
      p.size = (float)GetRandomValue(3, 9) / 2.f;
      p.type = type;
      particles.push_back(p);
    }
  }
  void emitTrail(Vector2 pos, Color col) {
    Particle p;
    p.pos = pos;
    float ang = (float)GetRandomValue(0, 628) / 100.f;
    float spd = (float)GetRandomValue(5, 40);
    p.vel = {cosf(ang) * spd, sinf(ang) * spd};
    p.col = col;
    p.life = (float)GetRandomValue(10, 30) / 100.f;
    p.maxLife = p.life;
    p.size = (float)GetRandomValue(2, 6) / 2.f;
    p.type = 2;
    particles.push_back(p);
  }
  void update(float dt) {
    for (int i = (int)particles.size() - 1; i >= 0; i--) {
      auto &p = particles[i];
      p.pos.x += p.vel.x * dt;
      p.pos.y += p.vel.y * dt;
      p.vel.y += 500.f * dt;
      p.vel.x *= 0.98f;
      p.life -= dt;
      if (p.life <= 0)
        particles.erase(particles.begin() + i);
    }
  }
  void draw() {
    for (auto &p : particles) {
      float t = p.life / p.maxLife;
      Color c = ColorAlpha(p.col, t);
      float s = p.size * t;
      if (p.type == 0)
        DrawRectanglePro({p.pos.x, p.pos.y, s * 2, s}, {s, s * 0.5f},
                         p.life * 200.f, c);
      else if (p.type == 1)
        DrawLineEx(p.pos,
                   {p.pos.x - p.vel.x * 0.05f, p.pos.y - p.vel.y * 0.05f}, s,
                   c);
      else
        DrawCircleV(p.pos, s, c);
    }
  }
};

// ============================================================
// SECTION 7: DRAWING PRIMITIVES
// ============================================================

// Beautiful glowing firefly circle
void DrawFirefly(Vector2 center, float baseR, Color col, float pulse,
                 bool isElite, float time) {
  // Outer corona
  float gR = baseR * (1.8f + 0.3f * sinf(pulse + time * 3.f));
  for (int ring = 5; ring >= 1; ring--) {
    float r = gR * ring / 5.f;
    float a = 0.04f * (float)(6 - ring) * (col.a / 255.f);
    DrawCircleV(center, r, ColorAlpha(col, a));
  }
  // Mid glow
  DrawCircleV(center, baseR * 1.2f, ColorAlpha(col, 0.25f));
  // Core
  DrawCircleV(center, baseR, col);
  // Specular highlight
  Vector2 hl = {center.x - baseR * 0.3f, center.y - baseR * 0.3f};
  DrawCircleV(hl, baseR * 0.25f, ColorAlpha(WHITE, 0.6f));

  if (isElite) {
    // Crown ring
    DrawRing(center, baseR * 1.3f, baseR * 1.5f, 0.f, 360.f, 32,
             ColorAlpha(PAL.gold, 0.7f + 0.3f * sinf(time * 4.f)));
    // Star spikes
    for (int s = 0; s < 6; s++) {
      float a = (float)s / 6.f * 6.283f + time;
      Vector2 sp = {center.x + cosf(a) * (baseR * 1.9f),
                    center.y + sinf(a) * (baseR * 1.9f)};
      DrawCircleV(sp, 2.f, PAL.gold);
    }
  }
}

// Draw Queen piece (improved)
void DrawQueenPiece(Vector2 center, float sz, Color col, bool isOptimal,
                    float time) {
  float r = sz * 0.36f;
  // Glow halo if optimal
  if (isOptimal) {
    for (int g = 5; g >= 1; g--)
      DrawCircleV(center, r * 1.4f * (float)g / 5.f,
                  ColorAlpha(PAL.gold, 0.05f * (float)(6 - g)));
  }
  // Body (orb)
  Color body = isOptimal ? PAL.gold : col;
  DrawCircleV(center, r, body);
  // Crown base
  float cx = center.x, cy = center.y;
  float bx = cx - r, by = cy;
  // Three prongs
  Vector2 prongs[3] = {{cx - r * 0.8f, cy - r * 1.6f},
                       {cx, cy - r * 2.0f},
                       {cx + r * 0.8f, cy - r * 1.6f}};
  DrawTriangle({cx - r, cy}, {cx + r, cy}, prongs[0], body);
  DrawTriangle({cx - r * 0.1f, cy - r * 0.3f}, {cx + r * 0.1f, cy - r * 0.3f},
               prongs[1], body);
  DrawTriangle({cx + r, cy}, prongs[2], {cx, cy}, body);
  for (auto &p : prongs)
    DrawCircleV(p, r * 0.22f, isOptimal ? WHITE : ColorBrightness(col, 0.4f));
  // Specular
  DrawCircleV({cx - r * 0.25f, cy - r * 0.25f}, r * 0.22f,
              ColorAlpha(WHITE, 0.55f));

  // Conflict indicators (diagonal attack lines shown on board separately)
}

// Panel with glowing border
void DrawPanel(Rectangle r, const char *title, Font font,
               Color borderCol = {35, 40, 65, 255}) {
  DrawRectangleRec(r, PAL.panel);
  DrawRectangleRoundedLines(r, 0.04f, 12, 1.0f, borderCol);
  if (title && title[0]) {
    // Subtle header separator
    DrawRectangle((int)r.x + 1, (int)r.y + 1, (int)r.width - 2, 30,
                  ColorAlpha(borderCol, 0.3f));
    DrawTextEx(font, title, {r.x + 12, r.y + 8}, 16, 1, PAL.text);
  }
}

// Axis + grid for graph
void DrawGraphAxes(Rectangle r, Font font, float maxVal, int steps) {
  // Grid lines
  for (int i = 0; i <= steps; i++) {
    float y = r.y + r.height - (float)i / steps * r.height;
    float val = maxVal * i / steps;
    DrawLineEx({r.x, y}, {r.x + r.width, y}, 1.f, ColorAlpha(PAL.axis, 0.4f));
    DrawTextEx(font, TextFormat("%.0f", val), {r.x - 30, y - 7}, 11, 1,
               PAL.dimText);
  }
  // Axes
  DrawLineEx({r.x, r.y}, {r.x, r.y + r.height}, 1.5f, PAL.axis);
  DrawLineEx({r.x, r.y + r.height}, {r.x + r.width, r.y + r.height}, 1.5f,
             PAL.axis);
}

// Smooth line graph for a series
void DrawSeries(Rectangle bounds, const std::vector<float> &data, int maxIter,
                float maxVal, Color col, float alpha = 1.f, bool fill = false) {
  if (data.size() < 2)
    return;
  float gw = bounds.width, gh = bounds.height;
  float ox = bounds.x, oy = bounds.y + gh;

  // Filled area
  if (fill && data.size() > 1) {
    for (int i = 0; i < (int)data.size() - 1; i++) {
      float x1 = ox + (float)i / maxIter * gw;
      float x2 = ox + (float)(i + 1) / maxIter * gw;
      float y1 = oy - (data[i] / maxVal) * gh;
      float y2 = oy - (data[i + 1] / maxVal) * gh;
      DrawTriangle({x1, oy}, {x1, y1}, {x2, y2}, ColorAlpha(col, 0.07f));
      DrawTriangle({x1, oy}, {x2, y2}, {x2, oy}, ColorAlpha(col, 0.07f));
    }
  }

  // Smooth polyline
  for (int i = 0; i < (int)data.size() - 1; i++) {
    Vector2 p1 = {ox + (float)i / maxIter * gw, oy - (data[i] / maxVal) * gh};
    Vector2 p2 = {ox + (float)(i + 1) / maxIter * gw,
                  oy - (data[i + 1] / maxVal) * gh};
    DrawLineEx(p1, p2, 2.f, ColorAlpha(col, alpha));
  }
  // Dot at end
  if (!data.empty()) {
    float ex = ox + (float)(data.size() - 1) / maxIter * gw;
    float ey = oy - (data.back() / maxVal) * gh;
    DrawCircleV({ex, ey}, 5.f, col);
    DrawCircleV({ex, ey}, 2.f, WHITE);
  }
}

// Heat map for queen positions
void DrawHeatMap(Rectangle bounds, const RunStats &stats, Font font,
                 int totalSamples) {
  float cw = bounds.width / N;
  float ch = bounds.height / N;
  // Find max for normalising
  int mx = 1;
  for (auto &row : stats.heatMap)
    for (int v : row)
      mx = std::max(mx, v);

  for (int r = 0; r < N; r++) {
    for (int c = 0; c < N; c++) {
      float t = (float)stats.heatMap[r][c] / mx;
      Color col =
          LerpColor(LerpColor(PAL.heatLow, PAL.heatMid, std::min(t * 2.f, 1.f)),
                    PAL.heatHigh, std::max(0.f, t * 2.f - 1.f));
      Rectangle cell = {bounds.x + c * cw, bounds.y + r * ch, cw - 1, ch - 1};
      DrawRectangleRec(cell, col);
      if (t > 0.5f) {
        DrawTextEx(font, TextFormat("%d%%", (int)(t * 100)),
                   {cell.x + 2, cell.y + 2}, 9, 1, ColorAlpha(WHITE, 0.7f));
      }
    }
  }
  // Col/row labels
  for (int i = 0; i < N; i++) {
    DrawTextEx(font, TextFormat("%c", 'A' + i),
               {bounds.x + i * cw + cw / 2 - 4, bounds.y + bounds.height + 3},
               11, 1, PAL.dimText);
    DrawTextEx(font, TextFormat("%d", N - i),
               {bounds.x - 16, bounds.y + i * ch + ch / 2 - 6}, 11, 1,
               PAL.dimText);
  }
}

// ============================================================
// SECTION 8: BUTTON
// ============================================================
struct Button {
  Rectangle rect;
  const char *label;
  Color col;
  float hoverT = 0.f;
  bool pressed = false;

  void update(float dt) {
    bool hover = CheckCollisionPointRec(GetMousePosition(), rect);
    hoverT = Lerp(hoverT, hover ? 1.f : 0.f, dt * 12.f);
    pressed = hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
  }
  void draw(Font font) {
    Color base = ColorBrightness(col, -0.3f + hoverT * 0.4f);
    Color border = LerpColor(col, WHITE, hoverT * 0.5f);
    DrawRectangleRounded(rect, 0.35f, 12, base);
    DrawRectangleRoundedLines(rect, 0.35f, 12, 1.5f, border);
    Vector2 ts = MeasureTextEx(font, label, 15, 1);
    DrawTextEx(font, label,
               {rect.x + rect.width / 2 - ts.x / 2,
                rect.y + rect.height / 2 - ts.y / 2},
               15, 1, WHITE);
  }
};

// ============================================================
// SECTION 9: TOOLTIP SYSTEM
// ============================================================
struct Tooltip {
  std::string text;
  Vector2 pos;
  float alpha = 0.f;
  bool visible = false;

  void show(const char *t, Vector2 p) {
    text = t;
    pos = p;
    visible = true;
  }
  void hide() { visible = false; }
  void update(float dt) {
    alpha = Lerp(alpha, visible ? 1.f : 0.f, dt * 10.f);
    visible = false;
  }
  void draw(Font font) {
    if (alpha < 0.01f)
      return;
    Vector2 sz = MeasureTextEx(font, text.c_str(), 13, 1);
    Rectangle r = {pos.x + 10, pos.y - 30, sz.x + 16, sz.y + 10};
    DrawRectangleRec(r, ColorAlpha({20, 22, 40, 255}, alpha));
    DrawRectangleLinesEx(r, 1, ColorAlpha(PAL.panelBord, alpha));
    DrawTextEx(font, text.c_str(), {r.x + 8, r.y + 5}, 13, 1,
               ColorAlpha(PAL.text, alpha));
  }
};

// ============================================================
// SECTION 10: LIVE STATS TICKER
// ============================================================
struct StatsTicker {
  struct Entry {
    std::string key, val;
    Color col;
  };
  std::vector<Entry> entries;
  void clear() { entries.clear(); }
  void add(const char *k, const char *v, Color c = {200, 210, 230, 255}) {
    entries.push_back({k, v, c});
  }
  void draw(Rectangle bounds, Font font) {
    float y = bounds.y + 8;
    float col1 = bounds.x + 10;
    float col2 = bounds.x + bounds.width * 0.5f;
    for (int i = 0;
         i < (int)entries.size() && y < bounds.y + bounds.height - 16; i++) {
      auto &e = entries[i];
      DrawTextEx(font, e.key.c_str(), {col1, y}, 13, 1, PAL.dimText);
      DrawTextEx(font, e.val.c_str(), {col2, y}, 13, 1, e.col);
      y += 20;
    }
  }
};

// ============================================================
// SECTION 11: BACKGROUND STARFIELD
// ============================================================
struct Star {
  Vector2 pos;
  float bright, speed;
};
struct Starfield {
  std::vector<Star> stars;
  void init(int count) {
    stars.resize(count);
    for (auto &s : stars) {
      s.pos = {(float)GetRandomValue(0, WIN_W),
               (float)GetRandomValue(0, WIN_H)};
      s.bright = (float)GetRandomValue(20, 100) / 100.f;
      s.speed = (float)GetRandomValue(5, 20) / 10.f;
    }
  }
  void update(float dt) {
    for (auto &s : stars) {
      s.bright += sinf(GetTime() * s.speed) * 0.01f;
      s.bright = Clamp(s.bright, 0.1f, 1.f);
    }
  }
  void draw() {
    for (auto &s : stars)
      DrawPixelV(s.pos, ColorAlpha({180, 200, 255, 255}, s.bright * 0.4f));
  }
};

// ============================================================
// SECTION 12: CONFLICT OVERLAY
// ============================================================
void DrawConflictLines(Rectangle boardRect, const std::vector<int> &pos,
                       float cell) {
  float bx = boardRect.x, by = boardRect.y;
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      if (std::abs(pos[i] - pos[j]) == std::abs(i - j)) {
        Vector2 p1 = {bx + pos[i] * cell + cell / 2, by + i * cell + cell / 2};
        Vector2 p2 = {bx + pos[j] * cell + cell / 2, by + j * cell + cell / 2};
        DrawLineEx(p1, p2, 2.f, ColorAlpha(PAL.conflict, 0.7f));
        DrawCircleV(p1, 4.f, ColorAlpha(PAL.conflict, 0.9f));
        DrawCircleV(p2, 4.f, ColorAlpha(PAL.conflict, 0.9f));
      }
    }
  }
}

// ============================================================
// SECTION 13: PROGRESS ARC
// ============================================================
void DrawProgressArc(Vector2 center, float r, float progress, Color col,
                     float time) {
  // Background ring
  DrawRing(center, r - 4, r, 0, 360, 64, ColorAlpha(PAL.panelBord, 0.5f));
  // Progress
  float endAngle = -90.f + progress * 360.f;
  if (endAngle > -90.f)
    DrawRing(center, r - 4, r, -90.f, endAngle, 64, col);
  // Glow dot at tip
  float tipA = (-90.f + progress * 360.f) * DEG2RAD;
  Vector2 tip = {center.x + cosf(tipA) * (r - 2.f),
                 center.y + sinf(tipA) * (r - 2.f)};
  DrawCircleV(tip, 6.f, col);
  DrawCircleV(tip, 3.f, WHITE);
}

// ============================================================
// SECTION 14: ANTENNA / FIREFLY VISUAL DETAILS
// ============================================================
void DrawFireflyDetailed(Vector2 center, float r, Color col, float pulse,
                         bool elite, float time, bool moving) {
  // Body shadow
  DrawCircleV({center.x + 2, center.y + 2}, r * 1.1f,
              ColorAlpha({0, 0, 0, 255}, 0.4f));

  // Wings (two ellipses)
  float wingSpread = r * 1.8f;
  float wingFlap = sinf(time * (moving ? 18.f : 6.f) + pulse) * 8.f;
  // Left wing
  DrawEllipse((int)(center.x - wingSpread * 0.6f),
              (int)(center.y - r * 0.3f + wingFlap * 0.3f), wingSpread * 0.5f,
              r * 0.6f, ColorAlpha(LerpColor(col, WHITE, 0.4f), 0.25f));
  // Right wing
  DrawEllipse((int)(center.x + wingSpread * 0.6f),
              (int)(center.y - r * 0.3f + wingFlap * 0.3f), wingSpread * 0.5f,
              r * 0.6f, ColorAlpha(LerpColor(col, WHITE, 0.4f), 0.25f));

  // Outer glow rings
  float gRad = r * (2.2f + 0.4f * sinf(pulse + time * 2.f));
  DrawCircleV(center, gRad, ColorAlpha(col, 0.06f));
  DrawCircleV(center, gRad * 0.7f, ColorAlpha(col, 0.12f));
  DrawCircleV(center, gRad * 0.45f, ColorAlpha(col, 0.22f));

  // Body
  DrawCircleV(center, r, col);
  // Abdomen glow (bioluminescent)
  Vector2 abd = {center.x, center.y + r * 0.5f};
  float glowPulse = 0.5f + 0.5f * sinf(pulse + time * 3.f);
  DrawCircleV(abd, r * 0.45f,
              ColorAlpha(LerpColor(col, PAL.gold, 0.5f), 0.8f * glowPulse));

  // Head
  DrawCircleV({center.x, center.y - r * 0.75f}, r * 0.32f,
              ColorBrightness(col, 0.2f));
  // Eyes
  DrawCircleV({center.x - r * 0.12f, center.y - r * 0.8f}, r * 0.1f, WHITE);
  DrawCircleV({center.x + r * 0.12f, center.y - r * 0.8f}, r * 0.1f, WHITE);
  DrawCircleV({center.x - r * 0.12f, center.y - r * 0.8f}, r * 0.05f, BLACK);
  DrawCircleV({center.x + r * 0.12f, center.y - r * 0.8f}, r * 0.05f, BLACK);

  // Antennae
  float antLen = r * 1.2f;
  DrawLineEx({center.x, center.y - r},
             {center.x - antLen * 0.7f, center.y - r - antLen}, 1.5f,
             ColorAlpha(col, 0.6f));
  DrawLineEx({center.x, center.y - r},
             {center.x + antLen * 0.7f, center.y - r - antLen}, 1.5f,
             ColorAlpha(col, 0.6f));
  DrawCircleV({center.x - antLen * 0.7f, center.y - r - antLen}, 2.5f, col);
  DrawCircleV({center.x + antLen * 0.7f, center.y - r - antLen}, 2.5f, col);

  // Specular
  DrawCircleV({center.x - r * 0.28f, center.y - r * 0.3f}, r * 0.18f,
              ColorAlpha(WHITE, 0.55f));

  // Elite crown
  if (elite) {
    float crownY = center.y - r * 1.8f;
    for (int s = 0; s < 5; s++) {
      float a = ((float)s / 5.f) * 6.283f - 1.571f;
      float cr = r * (s % 2 == 0 ? 1.4f : 0.9f);
      DrawCircleV({center.x + cosf(a) * cr, crownY + sinf(a) * r * 0.5f}, 2.5f,
                  PAL.gold);
    }
    DrawRing(center, r * 1.6f, r * 1.8f, 0, 360, 32,
             ColorAlpha(PAL.gold, 0.4f + 0.3f * sinf(time * 5.f)));
  }
}

// ============================================================
// SECTION 15: SWARM PANEL DRAW
// ============================================================
void DrawSwarmPanel(FAEngine &fa, float time, bool showTrails,
                    ParticleSystem &ps) {
  for (auto &ff : fa.pop) {
    // Emit trail particles occasionally
    if (showTrails && GetRandomValue(0, 10) == 0 &&
        Vector2Distance(ff.screenPos, ff.targetPos) > 3.f) {
      ps.emitTrail(ff.screenPos, ColorAlpha(ff.tint, 0.6f));
    }
    // Draw trail
    for (int t = 1; t < (int)ff.trail.pts.size(); t++) {
      float a = (float)t / ff.trail.pts.size();
      DrawLineEx(ff.trail.pts[t - 1], ff.trail.pts[t], 1.5f * a,
                 ColorAlpha(ff.tint, a * 0.4f));
    }
    // Attraction lines to nearest brighter neighbour
    float bestDist = 9999.f;
    Vector2 bestNeighbour = {};
    for (auto &other : fa.pop) {
      if (&other != &ff && other.fitness > ff.fitness) {
        float d = Vector2Distance(ff.screenPos, other.screenPos);
        if (d < bestDist) {
          bestDist = d;
          bestNeighbour = other.screenPos;
        }
      }
    }
    if (bestDist < 120.f) {
      DrawLineEx(ff.screenPos, bestNeighbour, 1.f,
                 ColorAlpha(ff.tint, 0.1f + 0.1f * (1.f - bestDist / 120.f)));
    }

    float r = 8.f + (ff.fitness / MAX_FIT) * 12.f;
    bool moving = Vector2Distance(ff.screenPos, ff.targetPos) > 5.f;
    DrawFireflyDetailed(ff.screenPos, r, ff.tint, ff.pulsePhase, ff.isElite,
                        time, moving);
  }
}

// ============================================================
// SECTION 16: ITERATION COMPARISON TABLE
// ============================================================
void DrawComparisonTable(Rectangle bounds, const FAEngine &orig,
                         const FAEngine &mod, Font font) {
  struct Row {
    const char *label;
    std::string valO, valM;
    bool highlight;
  };
  std::vector<Row> rows;

  auto fi = [](float f) { return TextFormat("%.1f", f); };
  auto ii = [](int i) { return i < 0 ? "—" : TextFormat("%d", i); };

  rows.push_back(
      {"Best Fitness", fi(orig.stats.bestFit), fi(mod.stats.bestFit), true});
  rows.push_back(
      {"Avg Fitness", fi(orig.stats.avgFit), fi(mod.stats.avgFit), false});
  rows.push_back({"Iter to Opt", ii(orig.stats.iterToOptimal),
                  ii(mod.stats.iterToOptimal), true});
  rows.push_back({"Total Iters", TextFormat("%d", orig.iter),
                  TextFormat("%d", mod.iter), false});
  rows.push_back({"Optimal Hits", TextFormat("%d", orig.stats.optimalCount),
                  TextFormat("%d", mod.stats.optimalCount), true});

  float rowH = 28.f;
  float y = bounds.y + 8;
  float xL = bounds.x + 10;
  float xO = bounds.x + bounds.width * 0.42f;
  float xM = bounds.x + bounds.width * 0.72f;

  // Header
  DrawTextEx(font, "Metric", {xL, y}, 13, 1, PAL.dimText);
  DrawTextEx(font, "Original", {xO, y}, 13, 1, PAL.orig);
  DrawTextEx(font, "Modified", {xM, y}, 13, 1, PAL.mod);
  y += rowH;
  DrawLineEx({bounds.x + 5, y - 4}, {bounds.x + bounds.width - 5, y - 4}, 1,
             PAL.panelBord);

  for (auto &row : rows) {
    if (row.highlight)
      DrawRectangleRec({bounds.x + 4, y - 2, bounds.width - 8, rowH - 4},
                       ColorAlpha(PAL.panelBord, 0.3f));
    DrawTextEx(font, row.label, {xL, y + 4}, 12, 1, PAL.text);
    DrawTextEx(font, row.valO.c_str(), {xO, y + 4}, 12, 1, PAL.orig);
    DrawTextEx(font, row.valM.c_str(), {xM, y + 4}, 12, 1, PAL.mod);
    y += rowH;
    if (y > bounds.y + bounds.height - 20)
      break;
  }
}

// ============================================================
// SECTION 17: SPEED SLIDER
// ============================================================
struct Slider {
  Rectangle track;
  float value = 0.5f; // 0..1
  float handleT = 0.f;
  bool dragging = false;

  void update() {
    Vector2 mp = GetMousePosition();
    float hx = track.x + value * track.width;
    bool hover =
        CheckCollisionPointCircle(mp, {hx, track.y + track.height / 2}, 14.f);
    handleT =
        Lerp(handleT, hover || dragging ? 1.f : 0.f, GetFrameTime() * 10.f);
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && hover)
      dragging = true;
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON))
      dragging = false;
    if (dragging)
      value = Clamp((mp.x - track.x) / track.width, 0.f, 1.f);
  }
  void draw(Font font, Color col) {
    // Track
    DrawRectangleRounded(track, 1.f, 8, ColorAlpha(PAL.axis, 0.5f));
    // Fill
    Rectangle fill = track;
    fill.width = value * track.width;
    DrawRectangleRounded(fill, 1.f, 8, ColorAlpha(col, 0.6f));
    // Handle
    float hx = track.x + value * track.width;
    float hy = track.y + track.height / 2;
    DrawCircleV({hx, hy}, 10.f + handleT * 3.f, col);
    DrawCircleV({hx, hy}, 5.f, WHITE);
    DrawTextEx(font, TextFormat("Speed: %.1fx", value * 5.f + 0.2f),
               {track.x + track.width + 12, hy - 8}, 14, 1, PAL.text);
  }
  float getInterval() { return Lerp(1.0f, 0.01f, value); }
};

// ============================================================
// SECTION 18: MAIN
// ============================================================
int main() {
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
  InitWindow(WIN_W, WIN_H, "Firefly Algorithm Visualizer — Next Level");
  SetTargetFPS(60);

  // Fonts
  Font fontBold = LoadFontEx("C:\\Windows\\Fonts\\arialbd.ttf", 28, 0, 250);
  Font fontMono = LoadFontEx("C:\\Windows\\Fonts\\consola.ttf", 24, 0, 250);
  Font fontSm = LoadFontEx("C:\\Windows\\Fonts\\arial.ttf", 20, 0, 250);
  // Fallback to default if not found
  if (fontBold.glyphCount <= 1)
    fontBold = GetFontDefault();
  if (fontMono.glyphCount <= 1)
    fontMono = GetFontDefault();
  if (fontSm.glyphCount <= 1)
    fontSm = GetFontDefault();

  // Shaders
  Shader bloomShader = LoadShaderFromMemory(nullptr, BLOOM_FRAG);
  Shader vignetteShader = LoadShaderFromMemory(nullptr, VIGNETTE_FRAG);
  int bloomResLoc = GetShaderLocation(bloomShader, "resolution");
  int bloomIntLoc = GetShaderLocation(bloomShader, "intensity");
  Vector2 res = {(float)WIN_W, (float)WIN_H};
  SetShaderValue(bloomShader, bloomResLoc, &res, SHADER_UNIFORM_VEC2);
  RenderTexture2D bloomTarget = LoadRenderTexture(WIN_W, WIN_H);

  // Systems
  Starfield starfield;
  starfield.init(300);
  ParticleSystem particles;
  Tooltip tooltip;
  StatsTicker ticker;
  Slider speedSlider;
  speedSlider.track = {1075, 840, 280, 8};
  speedSlider.value = 0.4f;

  // Engines
  FAEngine origFA, modFA;

  // Buttons
  Button btnOrig = {{20, 795, 140, 44}, "Original FA", PAL.orig};
  Button btnMod = {{170, 795, 140, 44}, "Modified FA", PAL.mod};
  Button btnBoth = {{320, 795, 140, 44}, "Run Both", PAL.accent};
  Button btnReset = {{470, 795, 120, 44}, "Reset", PAL.conflict};
  Button btnStep = {{600, 795, 100, 44}, "Step", PAL.safe};

  bool showO = false, showM = false, running = false;
  float stepTimer = 0.f;
  float time = 0.f;
  bool showConflicts = true;
  bool showTrails = true;
  bool showCmpTable = false;
  float celebTimer = 0.f;

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();
    time += dt;
    stepTimer += dt;

    // Button updates
    btnOrig.update(dt);
    btnMod.update(dt);
    btnBoth.update(dt);
    btnReset.update(dt);
    btnStep.update(dt);
    speedSlider.update();
    tooltip.update(dt);
    starfield.update(dt);
    particles.update(dt);

    if (btnOrig.pressed) {
      origFA.init(false, PAL.orig);
      showO = true;
      showM = false;
      running = true;
      showCmpTable = false;
      stepTimer = 0.f;
    }
    if (btnMod.pressed) {
      modFA.init(true, PAL.mod);
      showO = false;
      showM = true;
      running = true;
      showCmpTable = false;
      stepTimer = 0.f;
    }
    if (btnBoth.pressed) {
      origFA.init(false, PAL.orig);
      modFA.init(true, PAL.mod);
      showO = showM = true;
      running = true;
      showCmpTable = true;
      stepTimer = 0.f;
    }
    if (btnReset.pressed) {
      showO = showM = running = showCmpTable = false;
    }
    if (btnStep.pressed && (showO || showM)) {
      if (showO && !origFA.done)
        origFA.step();
      if (showM && !modFA.done)
        modFA.step();
    }

    // Simulation step
    if (running && stepTimer >= speedSlider.getInterval()) {
      bool any = false;
      if (showO && !origFA.done) {
        origFA.step();
        any = true;
      }
      if (showM && !modFA.done) {
        modFA.step();
        any = true;
      }
      stepTimer = 0.f;

      // Check completion
      bool odone = !showO || origFA.done;
      bool mdone = !showM || modFA.done;
      if (odone && mdone) {
        running = false;
        bool found = (showO && origFA.stats.bestFit >= MAX_FIT) ||
                     (showM && modFA.stats.bestFit >= MAX_FIT);
        if (found) {
          celebTimer = 3.f;
          particles.emit({WIN_W * 0.5f, WIN_H * 0.4f}, 120, 0, PAL.gold);
          particles.emit({WIN_W * 0.3f, WIN_H * 0.3f}, 80, 1, PAL.orig);
          particles.emit({WIN_W * 0.7f, WIN_H * 0.3f}, 80, 1, PAL.mod);
        }
      }
    }
    if (celebTimer > 0.f) {
      celebTimer -= dt;
      if (GetRandomValue(0, 4) == 0)
        particles.emit(
            {(float)GetRandomValue(200, 1400), (float)GetRandomValue(100, 600)},
            15, 0, PAL.gold);
    }

    // Firefly interpolation & trail
    auto updateSwarm = [&](FAEngine &fa) {
      for (auto &ff : fa.pop) {
        Vector2 prev = ff.screenPos;
        ff.screenPos = Vector2Lerp(ff.screenPos, ff.targetPos, 0.12f);
        ff.pulsePhase += dt * (2.f + ff.fitness / MAX_FIT * 4.f);
        ff.wobble += dt * 1.5f;
        if (Vector2Distance(prev, ff.screenPos) > 1.f)
          ff.trail.push(ff.screenPos);
      }
    };
    if (showO)
      updateSwarm(origFA);
    if (showM)
      updateSwarm(modFA);

    // Build ticker
    ticker.clear();
    if (showO) {
      auto b = origFA.getBest();
      ticker.add("Orig Best:", TextFormat("%d / %d", (int)b.fitness, MAX_FIT),
                 PAL.orig);
      ticker.add("Orig Iter:", TextFormat("%d / %d", origFA.iter, MAX_ITER),
                 PAL.dimText);
      if (origFA.stats.iterToOptimal >= 0)
        ticker.add("Orig->Opt:", TextFormat("%d", origFA.stats.iterToOptimal),
                   PAL.safe);
    }
    if (showM) {
      auto b = modFA.getBest();
      ticker.add("Mod Best:", TextFormat("%d / %d", (int)b.fitness, MAX_FIT),
                 PAL.mod);
      ticker.add("Mod Iter:", TextFormat("%d / %d", modFA.iter, MAX_ITER),
                 PAL.dimText);
      if (modFA.stats.iterToOptimal >= 0)
        ticker.add("Mod->Opt:", TextFormat("%d", modFA.stats.iterToOptimal),
                   PAL.safe);
    }

    // ====================================================
    // BLOOM TARGET: draw things that glow
    // ====================================================
    BeginTextureMode(bloomTarget);
    ClearBackground(BLANK);
    // Board glow squares for optimal queens
    if (showO && origFA.inited && origFA.stats.bestFit >= MAX_FIT) {
      auto b = origFA.getBest();
      float cell = BOARD_RECT.height / N;
      float bx = BOARD_RECT.x, by = BOARD_RECT.y;
      for (int r = 0; r < N; r++)
        DrawCircleV(
            {bx + b.position[r] * cell + cell / 2, by + r * cell + cell / 2},
            cell * 0.4f, ColorAlpha(PAL.gold, 0.3f));
    }
    if (showM && modFA.inited && modFA.stats.bestFit >= MAX_FIT) {
      auto b = modFA.getBest();
      float cell = BOARD_RECT.height / N;
      float bx = BOARD_RECT.x, by = BOARD_RECT.y;
      for (int r = 0; r < N; r++)
        DrawCircleV(
            {bx + b.position[r] * cell + cell / 2, by + r * cell + cell / 2},
            cell * 0.4f, ColorAlpha(PAL.mod, 0.2f));
    }
    // Swarm glow
    if (showO)
      for (auto &ff : origFA.pop) {
        DrawCircleV(ff.screenPos, (8.f + (ff.fitness / MAX_FIT) * 12.f) * 2.f,
                    ColorAlpha(PAL.orig, 0.08f));
      }
    if (showM)
      for (auto &ff : modFA.pop) {
        DrawCircleV(ff.screenPos, (8.f + (ff.fitness / MAX_FIT) * 12.f) * 2.f,
                    ColorAlpha(PAL.mod, 0.08f));
      }
    EndTextureMode();

    // ====================================================
    // MAIN DRAW
    // ====================================================
    BeginDrawing();
    ClearBackground(PAL.bg);

    // Background
    starfield.draw();
    // Subtle radial gradient overlay
    for (int i = 8; i >= 1; i--)
      DrawCircleV({WIN_W * 0.5f, WIN_H * 0.5f}, (float)i * 120.f,
                  ColorAlpha({20, 25, 60, 255}, 0.02f));

    // ---- TITLE BAR ----
    DrawRectangle(0, 0, WIN_W, 52, ColorAlpha({10, 12, 24, 255}, 0.95f));
    DrawTextEx(fontBold, "FIREFLY ALGORITHM VISUALIZER", {20, 12}, 24, 1,
               PAL.gold);
    DrawTextEx(fontSm, "N=8 Queens | FA Optimisation", {20, 36}, 14, 1,
               PAL.dimText);
    DrawTextEx(fontMono, TextFormat("%d FPS", GetFPS()), {WIN_W - 90.f, 16}, 16,
               1, PAL.safe);

    // Status pill
    const char *statusStr =
        running ? "RUNNING" : (showO || showM ? "PAUSED" : "IDLE");
    Color statusCol =
        running ? PAL.safe : (showO || showM ? PAL.gold : PAL.dimText);
    DrawRectangleRounded({WIN_W - 180.f, 10, 80, 22}, 0.5f, 8, statusCol);
    DrawRectangleRoundedLines({WIN_W - 180.f, 10, 80, 22}, 0.5f, 8, 1.0f,
                              statusCol);
    Vector2 stSz = MeasureTextEx(fontSm, statusStr, 12, 1);
    DrawTextEx(fontSm, statusStr, {WIN_W - 180.f + 40 - stSz.x / 2, 14}, 12, 1,
               statusCol);

    // ---- BOARD PANEL ----
    DrawPanel(BOARD_RECT, nullptr, fontSm, PAL.panelBord);
    // Coordinates
    float cell = BOARD_RECT.height / N;
    float bx = BOARD_RECT.x, by = BOARD_RECT.y;
    for (int c = 0; c < N; c++)
      DrawTextEx(fontSm, TextFormat("%c", 'A' + c),
                 {bx + c * cell + cell / 2 - 5, by - 20}, 14, 1, PAL.dimText);
    for (int r = 0; r < N; r++)
      DrawTextEx(fontSm, TextFormat("%d", N - r),
                 {bx - 18, by + r * cell + cell / 2 - 8}, 14, 1, PAL.dimText);
    // Squares
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        Color sq = (r + c) % 2 == 0 ? PAL.lightSq : PAL.darkSq;
        DrawRectangle((int)(bx + c * cell), (int)(by + r * cell), (int)cell,
                      (int)cell, sq);
        // Hover highlight
        if (CheckCollisionPointRec(GetMousePosition(),
                                   {bx + c * cell, by + r * cell, cell, cell}))
          DrawRectangle((int)(bx + c * cell), (int)(by + r * cell), (int)cell,
                        (int)cell, ColorAlpha(WHITE, 0.08f));
      }
    }
    // Conflict lines
    if (showConflicts) {
      if (showO && origFA.inited)
        DrawConflictLines(BOARD_RECT, origFA.getBest().position, cell);
      if (showM && modFA.inited && !showO)
        DrawConflictLines(BOARD_RECT, modFA.getBest().position, cell);
    }
    // Queens
    auto drawQueens = [&](FAEngine &fa, Color qcol) {
      if (!fa.inited)
        return;
      auto b = fa.getBest();
      bool opt = b.fitness >= MAX_FIT;
      for (int r = 0; r < N; r++) {
        Vector2 qpos = {bx + b.position[r] * cell + cell / 2,
                        by + r * cell + cell / 2};
        DrawQueenPiece(qpos, cell * 0.78f, qcol, opt, time);
      }
    };
    if (showO)
      drawQueens(origFA, PAL.orig);
    if (showM && !showO)
      drawQueens(modFA, PAL.mod);
    if (showO && showM) {
      // Draw both semi-transparently
      if (origFA.inited) {
        auto b = origFA.getBest();
        for (int r = 0; r < N; r++) {
          Vector2 qp = {bx + b.position[r] * cell + cell / 2,
                        by + r * cell + cell / 2};
          DrawQueenPiece(qp, cell * 0.72f, PAL.orig, b.fitness >= MAX_FIT,
                         time);
        }
      }
      if (modFA.inited) {
        auto b = modFA.getBest();
        for (int r = 0; r < N; r++) {
          Vector2 qp = {bx + b.position[r] * cell + cell / 2,
                        by + r * cell + cell / 2};
          DrawQueenPiece(qp, cell * 0.55f, PAL.mod, b.fitness >= MAX_FIT, time);
        }
      }
    }
    // Board border
    DrawRectangleLinesEx(BOARD_RECT, 2, PAL.panelBord);

    // Fitness label below board
    if (showO && origFA.inited) {
      auto b = origFA.getBest();
      DrawTextEx(fontSm, TextFormat("Orig: %d/%d", (int)b.fitness, MAX_FIT),
                 {bx, by + N * cell + 8}, 14, 1, PAL.orig);
    }
    if (showM && modFA.inited) {
      auto b = modFA.getBest();
      DrawTextEx(fontSm, TextFormat("Mod:  %d/%d", (int)b.fitness, MAX_FIT),
                 {bx + 150, by + N * cell + 8}, 14, 1, PAL.mod);
    }

    // ---- SWARM PANEL ----
    {
      Color swBorder = running ? LerpColor(PAL.panelBord, PAL.safe,
                                           0.5f + 0.5f * sinf(time * 4.f))
                               : PAL.panelBord;
      DrawPanel(SWARM_RECT,
                showM && !showO
                    ? "Modified FA — Swarm"
                    : (showO && !showM ? "Original FA — Swarm" : "Dual Swarm"),
                fontSm, swBorder);
      BeginScissorMode((int)SWARM_RECT.x, (int)SWARM_RECT.y,
                       (int)SWARM_RECT.width, (int)SWARM_RECT.height);
      if (showO)
        DrawSwarmPanel(origFA, time, showTrails, particles);
      if (showM)
        DrawSwarmPanel(modFA, time, showTrails, particles);
      EndScissorMode();

      // Iter progress arcs (top right of swarm panel)
      float arcR = 22.f;
      Vector2 arcO = {SWARM_RECT.x + SWARM_RECT.width - 70, SWARM_RECT.y + 50};
      Vector2 arcM = {SWARM_RECT.x + SWARM_RECT.width - 25, SWARM_RECT.y + 50};
      if (showO)
        DrawProgressArc(arcO, arcR,
                        origFA.inited ? (float)origFA.iter / MAX_ITER : 0.f,
                        PAL.orig, time);
      if (showM)
        DrawProgressArc(arcM, arcR,
                        modFA.inited ? (float)modFA.iter / MAX_ITER : 0.f,
                        PAL.mod, time);
    }

    // ---- CONVERGENCE GRAPH ----
    {
      DrawPanel(GRAPH_RECT, "Convergence Graph", fontSm);
      Rectangle gInner = {GRAPH_RECT.x + 44, GRAPH_RECT.y + 34,
                          GRAPH_RECT.width - 58, GRAPH_RECT.height - 46};
      DrawGraphAxes(gInner, fontSm, (float)MAX_FIT, 4);
      // Iter labels
      for (int i = 0; i <= 4; i++) {
        float x = gInner.x + (float)i / 4 * gInner.width;
        DrawTextEx(fontSm, TextFormat("%d", (int)(MAX_ITER * i / 4.f)),
                   {x - 8, gInner.y + gInner.height + 4}, 11, 1, PAL.dimText);
      }
      if (showO) {
        DrawSeries(gInner, origFA.stats.worstPerIter, MAX_ITER, MAX_FIT,
                   PAL.origDim, 0.5f, false);
        DrawSeries(gInner, origFA.stats.avgPerIter, MAX_ITER, MAX_FIT, PAL.orig,
                   0.7f, false);
        DrawSeries(gInner, origFA.stats.bestPerIter, MAX_ITER, MAX_FIT,
                   PAL.orig, 1.0f, true);
      }
      if (showM) {
        DrawSeries(gInner, modFA.stats.worstPerIter, MAX_ITER, MAX_FIT,
                   PAL.modDim, 0.5f, false);
        DrawSeries(gInner, modFA.stats.avgPerIter, MAX_ITER, MAX_FIT, PAL.mod,
                   0.7f, false);
        DrawSeries(gInner, modFA.stats.bestPerIter, MAX_ITER, MAX_FIT, PAL.mod,
                   1.0f, true);
      }
      // Legend
      DrawCircleV({gInner.x + gInner.width - 100, gInner.y - 12}, 5, PAL.orig);
      DrawTextEx(fontSm, "Best", {gInner.x + gInner.width - 92, gInner.y - 19},
                 12, 1, PAL.orig);
      DrawCircleV({gInner.x + gInner.width - 55, gInner.y - 12}, 5, PAL.mod);
      DrawTextEx(fontSm, "Mod", {gInner.x + gInner.width - 47, gInner.y - 19},
                 12, 1, PAL.mod);
    }

    // ---- METRICS / COMPARISON PANEL ----
    {
      DrawPanel(METRICS_RECT,
                showCmpTable ? "Algorithm Comparison" : "Live Metrics", fontSm);
      if (showCmpTable && showO && showM && origFA.inited && modFA.inited) {
        Rectangle tbl = {METRICS_RECT.x + 4, METRICS_RECT.y + 32,
                         METRICS_RECT.width - 8, METRICS_RECT.height - 36};
        DrawComparisonTable(tbl, origFA, modFA, fontSm);
      } else {
        // Individual stats
        Rectangle tkR = {METRICS_RECT.x, METRICS_RECT.y + 32,
                         METRICS_RECT.width, METRICS_RECT.height - 36};
        ticker.draw(tkR, fontSm);

        // Fitness bar
        float barY = METRICS_RECT.y + METRICS_RECT.height - 80;
        if (showO && origFA.inited) {
          float ft = origFA.getBest().fitness / MAX_FIT;
          DrawRectangle((int)(METRICS_RECT.x + 10), (int)barY, 200, 16,
                        ColorAlpha(PAL.panelBord, 0.6f));
          DrawRectangle((int)(METRICS_RECT.x + 10), (int)barY, (int)(200 * ft),
                        16, PAL.orig);
          DrawTextEx(fontSm, "Orig Fitness", {METRICS_RECT.x + 10, barY - 16},
                     12, 1, PAL.dimText);
        }
        barY += 36;
        if (showM && modFA.inited) {
          float ft = modFA.getBest().fitness / MAX_FIT;
          DrawRectangle((int)(METRICS_RECT.x + 10), (int)barY, 200, 16,
                        ColorAlpha(PAL.panelBord, 0.6f));
          DrawRectangle((int)(METRICS_RECT.x + 10), (int)barY, (int)(200 * ft),
                        16, PAL.mod);
          DrawTextEx(fontSm, "Mod Fitness", {METRICS_RECT.x + 10, barY - 16},
                     12, 1, PAL.dimText);
        }
      }
    }

    // ---- HEAT MAP PANEL ----
    {
      DrawPanel(HEATMAP_RECT, "Queen Position Heat Map", fontSm);
      bool hasData = (showO && origFA.inited) || (showM && modFA.inited);
      if (hasData) {
        // Merge heat maps
        RunStats merged;
        merged.heatMap = {};
        if (showO && origFA.inited)
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
              merged.heatMap[r][c] += origFA.stats.heatMap[r][c];
        if (showM && modFA.inited)
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
              merged.heatMap[r][c] += modFA.stats.heatMap[r][c];
        Rectangle hmInner = {HEATMAP_RECT.x + 24, HEATMAP_RECT.y + 36,
                             HEATMAP_RECT.width - 36, HEATMAP_RECT.height - 52};
        DrawHeatMap(hmInner, merged, fontSm, 1);
      } else {
        Vector2 ts =
            MeasureTextEx(fontSm, "Run an algorithm to populate", 13, 1);
        DrawTextEx(fontSm, "Run an algorithm to populate",
                   {HEATMAP_RECT.x + HEATMAP_RECT.width / 2 - ts.x / 2,
                    HEATMAP_RECT.y + HEATMAP_RECT.height / 2 - 8},
                   13, 1, PAL.dimText);
      }
    }

    // ---- CONTROL PANEL ----
    DrawRectangleRounded(CTRL_RECT, 0.08f, 12, ColorAlpha(PAL.panel, 0.9f));
    DrawRectangleRoundedLines(CTRL_RECT, 0.08f, 12, 2.0f, PAL.panelBord);
    btnOrig.draw(fontBold);
    btnMod.draw(fontBold);
    btnBoth.draw(fontBold);
    btnReset.draw(fontBold);
    btnStep.draw(fontBold);
    speedSlider.draw(fontSm, PAL.gold);

    // Checkboxes for options
    auto drawCheck = [&](float x, float y, const char *label, bool &val) {
      Rectangle cr = {x, y, 16, 16};
      bool hover = CheckCollisionPointRec(GetMousePosition(), cr);
      if (hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        val = !val;
      DrawRectangleRec(cr, val ? PAL.safe : ColorAlpha(PAL.panelBord, 0.5f));
      DrawRectangleLinesEx(cr, 1, val ? PAL.safe : PAL.dimText);
      if (val)
        DrawTextEx(fontSm, "✓", {x + 2, y + 1}, 13, 1, BLACK);
      DrawTextEx(fontSm, label, {x + 22, y + 1}, 13, 1, PAL.text);
    };
    drawCheck(730, 800, "Conflict Lines", showConflicts);
    drawCheck(730, 820, "Trails", showTrails);
    drawCheck(850, 800, "Compare Table", showCmpTable);

    // Particles
    particles.draw();

    // Bloom overlay
    float bloomInt = 1.8f;
    SetShaderValue(bloomShader, bloomIntLoc, &bloomInt, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(bloomShader);
    DrawTextureRec(bloomTarget.texture, {0, 0, (float)WIN_W, -(float)WIN_H},
                   {0, 0}, ColorAlpha(WHITE, 0.6f));
    EndShaderMode();

    // Tooltip
    tooltip.draw(fontSm);

    // Celebration overlay
    if (celebTimer > 0.f) {
      float a = std::min(celebTimer, 1.f);
      DrawTextEx(fontBold, "SOLUTION FOUND!",
                 {WIN_W / 2.f - 150, WIN_H / 2.f - 20}, 36, 2,
                 ColorAlpha(PAL.gold, a));
    }

    EndDrawing();
  }

  UnloadShader(bloomShader);
  UnloadShader(vignetteShader);
  UnloadRenderTexture(bloomTarget);
  UnloadFont(fontBold);
  UnloadFont(fontMono);
  UnloadFont(fontSm);
  CloseWindow();
  return 0;
}
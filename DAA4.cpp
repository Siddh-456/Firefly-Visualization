// ============================================================
//  FIREFLY ALGORITHM VISUALIZER — v2 (Parameter Tuning + Comparison)
//  Raylib 5.x | C++17
//  Build: g++ firefly_visualizer_v2.cpp -o fa_viz -lraylib -lGL -lm -lpthread
//  -ldl -lrt -lX11 Or on Windows with raylib installed: g++
//  firefly_visualizer_v2.cpp -o fa_viz.exe -lraylib -lopengl32 -lgdi32 -lwinmm
// ============================================================

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
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
static int N = 8;
static int POP = 24;
static int MAX_ITER = 120;
static int MAX_FIT = N * (N - 1) / 2; // 28

static void recalcParams() {
  MAX_FIT = N * (N - 1) / 2;
  if (N <= 10) {
    POP = 24;
    MAX_ITER = 120;
  } else if (N <= 16) {
    POP = std::min(80, 16 + N * 4);
    MAX_ITER = std::min(1500, 100 + N * 60);
  } else {
    POP = 80;
    MAX_ITER = 1500;
  }
}

static const Rectangle BOARD_RECT = {42, 85, 460, 460};
static const Rectangle SWARM_RECT = {520, 60, 540, 320};
static const Rectangle GRAPH_RECT = {520, 390, 540, 360};
static const Rectangle METRICS_RECT = {1075, 60, 500, 320};
static const Rectangle HEATMAP_RECT = {1075, 390, 500, 360};
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

  Color orig = {60, 140, 255, 255};
  Color origDim = {30, 70, 130, 255};
  Color mod = {255, 130, 30, 255};
  Color modDim = {130, 65, 15, 255};

  Color gold = {255, 220, 40, 255};
  Color goldGlow = {255, 190, 0, 100};
  Color conflict = {255, 50, 60, 255};
  Color safe = {60, 220, 90, 255};
  Color accent = {180, 80, 255, 255};

  Color heatLow = {15, 20, 50, 255};
  Color heatMid = {100, 40, 200, 255};
  Color heatHigh = {255, 180, 40, 255};

  // Comparison colours
  Color improved = {50, 230, 120, 255};
  Color worsened = {255, 70, 80, 255};
  Color neutral = {180, 180, 180, 255};
};

static Palette PAL;

// ============================================================
// SECTION 4: UTILITY
// ============================================================
static Color LerpColor(Color a, Color b, float t) {
  return {(unsigned char)Lerp(a.r, b.r, t), (unsigned char)Lerp(a.g, b.g, t),
          (unsigned char)Lerp(a.b, b.b, t), (unsigned char)Lerp(a.a, b.a, t)};
}
static Color FitnessColor(float f) {
  float t = f / MAX_FIT;
  if (t < 0.5f)
    return LerpColor(PAL.conflict, PAL.mod, t * 2.f);
  return LerpColor(PAL.mod, PAL.gold, (t - 0.5f) * 2.f);
}
static void DrawRoundedBorder(Rectangle r, float round, int segs, float thick,
                              Color c) {
  DrawRectangleRoundedLines(r, round, segs, thick, c);
}
static float hash21(float x, float y) {
  float h = x * 127.1f + y * 311.7f;
  return fmodf(sinf(h) * 43758.5453f, 1.f);
}

// ============================================================
// SECTION 5: USER-TUNABLE MODIFICATION PARAMETERS
// ============================================================

// Stores the user-chosen values for the Modified FA
struct ModParams {
  // 1. Heuristic Init ratio: 0.0 = fully random, 1.0 = fully heuristic
  float heuristicRatio = 1.0f; // default: full heuristic (original behaviour)

  // 2. Alpha0: initial randomness coefficient
  float alpha0 = 0.9f; // default: Modified FA default

  // 3. Mutation rate: 0.0 to 1.0
  float mutationRate = 0.20f; // default: 0.20 (original Modified behaviour)

  // 4. Elite count: 0 to 4
  int eliteCount = 2; // default: top-2 (original Modified behaviour)

  // Labels for display
  static const char *heuristicLabel(float v) {
    if (v < 0.25f)
      return "Mostly Random";
    if (v < 0.5f)
      return "Partial Heuristic";
    if (v < 0.75f) return "Balanced";
    if (v < 0.95f)
      return "Mostly Heuristic";
    return "Full Heuristic";
  }
};

// ============================================================
// SECTION 6: ALGORITHM ENGINE
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
      if (mi < (int)miss.size())
        pos[i] = miss[mi++];
      else
        pos[i] = (int)(rng() % N);
      continue;
    }
    if (cnt[pos[i]] > 1) {
      cnt[pos[i]]--;
      if (mi < (int)miss.size())
        pos[i] = miss[mi++];
      else
        pos[i] = (int)(rng() % N);
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
  std::vector<std::vector<int>> heatMap;
};

// Post-run comparison result
struct ComparisonResult {
  bool computed = false;
  float origBest = 0.f;
  float modBest = 0.f;
  int origIterOpt = -1;
  int modIterOpt = -1;
  int origOptHits = 0;
  int modOptHits = 0;
  float origAvg = 0.f;
  float modAvg = 0.f;
  // Delta values (mod - orig), positive = improved
  float deltaFit = 0.f;
  int deltaIter = 0; // negative = mod was faster (better)
  int deltaHits = 0;
  float deltaAvg = 0.f;
  // Verdicts
  bool fitImproved = false;
  bool iterImproved = false;
  bool hitsImproved = false;
  bool avgImproved = false;
  int score = 0; // 0..4: how many metrics improved

  // User param record (what was used for this comparison)
  ModParams usedParams;
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
  float prevBest = 0.f;
  int stagnantCount = 0;
  float diversity = 0.f;
  std::vector<float> diversityPerIter;
  int swarmHalf = 0;
  int userEliteCount = 2;     // user-overridable
  float userMutRate = 0.20f;  // user-overridable
  float userHeurRatio = 1.0f; // user-overridable

  std::uniform_real_distribution<float> mutDist{0.f, 1.f};
  std::uniform_int_distribution<int> rowDist{0, 1};

  void init(bool modified, Color c, ModParams *mp = nullptr) {
    std::random_device rd;
    rng.seed(rd());
    iter = 0;
    done = false;
    inited = true;
    prevBest = 0.f;
    stagnantCount = 0;
    diversity = 0.f;
    diversityPerIter.clear();
    isModified = modified;
    col = c;
    stats = RunStats{};
    adaptive = elitism = heurInit = modified;

    if (modified && mp) {
      alpha0 = mp->alpha0;
      userEliteCount = mp->eliteCount;
      userMutRate = mp->mutationRate;
      userHeurRatio = mp->heuristicRatio;
    } else if (modified) {
      alpha0 = 0.9f;
      userEliteCount = 2;
      userMutRate = 0.20f;
      userHeurRatio = 1.0f;
    } else {
      alpha0 = 0.5f;
      userEliteCount = 0;
      userMutRate = 0.f;
      userHeurRatio = 0.f;
    }
    alpha = alpha0;
    rowDist = std::uniform_int_distribution<int>{0, N - 1};
    stats.heatMap.assign(N, std::vector<int>(N, 0));
    std::vector<int> base(N);
    std::iota(base.begin(), base.end(), 0);
    pop.resize(POP);

    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (auto &ff : pop) {
      bool useHeur = modified && (u01(rng) < userHeurRatio);
      ff.position = useHeur ? heuristicInit(rng) : base;
      if (!useHeur)
        std::shuffle(ff.position.begin(), ff.position.end(), rng);
      ff.fitness = (float)calcFitness(ff.position);
      ff.trail.clear();
      std::uniform_int_distribution<int> phase(0, 628);
      ff.pulsePhase = (float)phase(rng) / 100.f;
      ff.wobble = (float)phase(rng) / 100.f;
      ff.tint = col;
    }
    layoutSwarm();
    for (auto &ff : pop)
      ff.screenPos = ff.targetPos;
  }

  void layoutSwarm(int half = 0) {
    float sx = SWARM_RECT.x + 14.f;
    float sy = SWARM_RECT.y + 40.f;
    float pw = SWARM_RECT.width - 28.f;
    float ph = SWARM_RECT.height - 86.f;
    if (half == 1) {
      pw = pw / 2.f;
    }
    if (half == 2) {
      sx += (SWARM_RECT.width - 28.f) / 2.f;
      pw = pw / 2.f;
    }
    const int cols = half != 0 ? 3 : 6;
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

    // Elitism — cache top userEliteCount fireflies
    std::vector<Firefly> elites;
    if (elitism && userEliteCount > 0) {
      for (auto &ff : pop)
        ff.isElite = false;
      auto sorted = pop;
      std::sort(sorted.begin(), sorted.end(),
                [](auto &a, auto &b) { return a.fitness > b.fitness; });
      for (int e = 0; e < userEliteCount && e < (int)sorted.size(); e++) {
        elites.push_back(sorted[e]);
        for (auto &ff : pop)
          if (ff.position == sorted[e].position) {
            ff.isElite = true;
            break;
          }
      }
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

    // Discrete swap mutation — uses user-defined rate
    if (isModified && userMutRate > 0.f) {
      int nSwaps = std::max(1, (N - 6) / 3);
      float progress = (float)iter / MAX_ITER;
      float alphaNorm = alpha0 / std::max(1.f, sqrtf((float)N));
      alpha = alphaNorm * (1.f - progress * 0.8f) + 0.02f;
      for (int i = 0; i < POP; i++) {
        if (mutDist(rng) < userMutRate) {
          for (int s = 0; s < nSwaps; s++) {
            int r1 = rowDist(rng);
            int r2 = rowDist(rng);
            std::swap(pop[i].position[r1], pop[i].position[r2]);
          }
          pop[i].fitness = (float)calcFitness(pop[i].position);
        }
      }
    }

    std::sort(pop.begin(), pop.end(),
              [](auto &a, auto &b) { return a.fitness > b.fitness; });

    // Elitism — reinsert cached elites at tail
    if (elitism && !elites.empty()) {
      for (int e = 0; e < (int)elites.size(); e++)
        pop[POP - 1 - e] = elites[e];
    }

    float best = 0.f, worst = (float)(MAX_FIT + 1), avg = 0.f;
    bool hasOptThisIter = false;
    for (auto &ff : pop) {
      if (ff.fitness > best)
        best = ff.fitness;
      if (ff.fitness < worst)
        worst = ff.fitness;
      avg += ff.fitness;
      if (ff.fitness >= (float)MAX_FIT)
        hasOptThisIter = true;
      for (int r = 0; r < N; r++)
        if (ff.position[r] >= 0 && ff.position[r] < N)
          stats.heatMap[r][ff.position[r]]++;
    }
    avg /= POP;
    if (hasOptThisIter)
      stats.optimalCount++;
    stats.bestFit = std::max(stats.bestFit, best);
    stats.avgFit = avg;
    stats.bestPerIter.push_back(best);
    stats.avgPerIter.push_back(avg);
    stats.worstPerIter.push_back(worst);
    if (stats.iterToOptimal < 0 && best >= (float)MAX_FIT)
      stats.iterToOptimal = iter;

    if (best > prevBest) {
      prevBest = best;
      stagnantCount = 0;
    } else {
      stagnantCount++;
    }
    if (stagnantCount >= 40 && best < (float)MAX_FIT) {
      stagnantCount = 0;
      float bestRatio = best / MAX_FIT;
      std::vector<int> base(N);
      std::iota(base.begin(), base.end(), 0);
      if (bestRatio > 0.85f) {
        for (int i = 3 * POP / 4; i < POP; i++) {
          bool swapped = false;
          for (int r1 = 0; r1 < N && !swapped; r1++) {
            for (int r2 = r1 + 1; r2 < N && !swapped; r2++) {
              if (std::abs(pop[i].position[r1] - pop[i].position[r2]) ==
                  std::abs(r1 - r2)) {
                int r3 = rowDist(rng);
                std::swap(pop[i].position[r1], pop[i].position[r3]);
                float nf = (float)calcFitness(pop[i].position);
                if (nf >= pop[i].fitness) {
                  pop[i].fitness = nf;
                  swapped = true;
                } else
                  std::swap(pop[i].position[r1], pop[i].position[r3]);
              }
            }
          }
        }
      } else {
        std::uniform_real_distribution<float> u01b(0.f, 1.f);
        for (int i = POP / 2; i < POP; i++) {
          bool useHeur = isModified && (u01b(rng) < userHeurRatio);
          if (useHeur)
            pop[i].position = heuristicInit(rng);
          else {
            pop[i].position = base;
            std::shuffle(pop[i].position.begin(), pop[i].position.end(), rng);
          }
          pop[i].fitness = (float)calcFitness(pop[i].position);
        }
      }
    }

    {
      float total = 0.f;
      int pairs = 0;
      for (int i = 0; i < POP; i++)
        for (int j = i + 1; j < POP; j++) {
          int diff = 0;
          for (int k = 0; k < N; k++)
            if (pop[i].position[k] != pop[j].position[k])
              diff++;
          total += (float)diff / N;
          pairs++;
        }
      diversity = pairs > 0 ? total / pairs : 0.f;
      diversityPerIter.push_back(diversity);
    }

    layoutSwarm(swarmHalf);
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
// BOARD ANIMATOR
// ============================================================
struct BoardAnimator {
  std::vector<Vector2> queenScreenPos;
  std::vector<Vector2> queenTargetPos;
  std::vector<std::deque<Vector2>> trails;
  static const int TRAIL_MAX = 12;
  bool inited = false;

  void init(int n, Rectangle boardRect) {
    float cell = boardRect.height / n;
    queenScreenPos.assign(n, {boardRect.x + boardRect.width / 2.f,
                              boardRect.y + boardRect.height / 2.f});
    queenTargetPos.assign(n, {0, 0});
    trails.assign(n, {});
    inited = true;
  }

  void setTargets(const std::vector<int> &pos, int n, Rectangle boardRect) {
    float cell = boardRect.width / n;
    for (int r = 0; r < n; r++) {
      queenTargetPos[r] = {boardRect.x + pos[r] * cell + cell * 0.5f,
                           boardRect.y + r * cell + cell * 0.5f};
    }
  }

  void update(float dt, float speed = 9.f) {
    for (int i = 0; i < (int)queenScreenPos.size(); i++) {
      Vector2 prev = queenScreenPos[i];
      queenScreenPos[i] =
          Vector2Lerp(queenScreenPos[i], queenTargetPos[i], dt * speed);
      if (Vector2Distance(prev, queenTargetPos[i]) > 2.f) {
        trails[i].push_back(queenScreenPos[i]);
        if ((int)trails[i].size() > TRAIL_MAX)
          trails[i].pop_front();
      }
    }
  }

  void drawTrails(Color col) {
    for (int i = 0; i < (int)trails.size(); i++) {
      for (int t = 1; t < (int)trails[i].size(); t++) {
        float a = (float)t / trails[i].size();
        DrawLineEx(trails[i][t - 1], trails[i][t], 3.f * a,
                   ColorAlpha(col, a * 0.4f));
      }
    }
  }
};

// ============================================================
// SECTION 7: PARTICLE SYSTEM
// ============================================================
struct Particle {
  Vector2 pos, vel;
  Color col;
  float life, maxLife, size;
  int type;
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
// SECTION 8: DRAWING PRIMITIVES
// ============================================================
void DrawFireflyDetailed(Vector2 center, float r, Color col, float pulse,
                         bool elite, float time, bool moving) {
  DrawCircleV({center.x + 2, center.y + 2}, r * 1.1f,
              ColorAlpha({0, 0, 0, 255}, 0.4f));
  float wingSpread = r * 1.8f;
  float wingFlap = sinf(time * (moving ? 18.f : 6.f) + pulse) * 8.f;
  DrawEllipse((int)(center.x - wingSpread * 0.6f),
              (int)(center.y - r * 0.3f + wingFlap * 0.3f), wingSpread * 0.5f,
              r * 0.6f, ColorAlpha(LerpColor(col, WHITE, 0.4f), 0.25f));
  DrawEllipse((int)(center.x + wingSpread * 0.6f),
              (int)(center.y - r * 0.3f + wingFlap * 0.3f), wingSpread * 0.5f,
              r * 0.6f, ColorAlpha(LerpColor(col, WHITE, 0.4f), 0.25f));
  float gRad = r * (2.2f + 0.4f * sinf(pulse + time * 2.f));
  DrawCircleV(center, gRad * 0.6f, ColorAlpha(col, 0.05f));
  DrawCircleV(center, gRad * 0.35f, ColorAlpha(col, 0.10f));
  DrawCircleV(center, r, col);
  Vector2 abd = {center.x, center.y + r * 0.5f};
  float glowPulse = 0.5f + 0.5f * sinf(pulse + time * 3.f);
  DrawCircleV(abd, r * 0.45f,
              ColorAlpha(LerpColor(col, PAL.gold, 0.5f), 0.8f * glowPulse));
  DrawCircleV({center.x, center.y - r * 0.75f}, r * 0.32f,
              ColorBrightness(col, 0.2f));
  DrawCircleV({center.x - r * 0.12f, center.y - r * 0.8f}, r * 0.1f, WHITE);
  DrawCircleV({center.x + r * 0.12f, center.y - r * 0.8f}, r * 0.1f, WHITE);
  DrawCircleV({center.x - r * 0.12f, center.y - r * 0.8f}, r * 0.05f, BLACK);
  DrawCircleV({center.x + r * 0.12f, center.y - r * 0.8f}, r * 0.05f, BLACK);
  float antLen = r * 1.2f;
  DrawLineEx({center.x, center.y - r},
             {center.x - antLen * 0.7f, center.y - r - antLen}, 1.5f,
             ColorAlpha(col, 0.6f));
  DrawLineEx({center.x, center.y - r},
             {center.x + antLen * 0.7f, center.y - r - antLen}, 1.5f,
             ColorAlpha(col, 0.6f));
  DrawCircleV({center.x - antLen * 0.7f, center.y - r - antLen}, 2.5f, col);
  DrawCircleV({center.x + antLen * 0.7f, center.y - r - antLen}, 2.5f, col);
  DrawCircleV({center.x - r * 0.28f, center.y - r * 0.3f}, r * 0.18f,
              ColorAlpha(WHITE, 0.55f));
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

void DrawQueenPiece(Vector2 center, float sz, Color col, bool isOptimal,
                    float time) {
  (void)time;
  float r = sz * 0.42f;
  float vc = r * 0.28f;
  center.y += vc;
  DrawEllipse((int)(center.x + 2), (int)(center.y + r * 1.05f + 2),
              (int)(r * 1.1f), (int)(r * 0.22f),
              ColorAlpha({0, 0, 0, 255}, 0.35f));
  Color body = col;
  Color dark = ColorBrightness(body, -0.28f);
  Color light = ColorBrightness(body, 0.32f);
  Color outline = ColorBrightness(body, -0.55f);
  float rimW = r * 1.18f, rimH = r * 0.20f, rimY = center.y + r * 0.82f;
  DrawRectangleRounded({center.x - rimW, rimY, rimW * 2.f, rimH}, 0.5f, 8,
                       dark);
  DrawRectangleRoundedLines({center.x - rimW, rimY, rimW * 2.f, rimH}, 0.5f, 8,
                            1.2f, outline);
  float rim2W = r * 1.0f, rim2H = r * 0.16f, rim2Y = rimY - rim2H + 2;
  DrawRectangleRounded({center.x - rim2W, rim2Y, rim2W * 2.f, rim2H}, 0.5f, 8,
                       ColorBrightness(dark, 0.08f));
  DrawRectangleRoundedLines({center.x - rim2W, rim2Y, rim2W * 2.f, rim2H}, 0.5f,
                            8, 1.0f, outline);
  float skirtTopW = r * 0.68f, skirtBotW = r * 0.98f,
        skirtTopY = center.y + r * 0.10f, skirtBotY = rim2Y + rim2H * 0.5f;
  Vector2 sk[4] = {{center.x - skirtTopW, skirtTopY},
                   {center.x + skirtTopW, skirtTopY},
                   {center.x + skirtBotW, skirtBotY},
                   {center.x - skirtBotW, skirtBotY}};
  DrawTriangle(sk[3], sk[0], sk[1], body);
  DrawTriangle(sk[3], sk[1], sk[2], body);
  DrawLineEx(sk[0], sk[3], 1.3f, outline);
  DrawLineEx(sk[1], sk[2], 1.3f, outline);
  DrawLineEx(sk[2], sk[3], 1.3f, outline);
  for (int line = 1; line <= 2; line++) {
    float t2 = (float)line / 3.f, ly = skirtTopY + (skirtBotY - skirtTopY) * t2,
          lw = skirtTopW + (skirtBotW - skirtTopW) * t2;
    DrawLineEx({center.x - lw, ly}, {center.x + lw, ly}, 1.0f,
               ColorAlpha(outline, 0.55f));
  }
  float bandH = r * 0.22f, bandW = r * 0.70f,
        bandY = skirtTopY - bandH + r * 0.04f;
  DrawRectangleRounded({center.x - bandW, bandY, bandW * 2.f, bandH}, 0.3f, 8,
                       body);
  DrawRectangleRoundedLines({center.x - bandW, bandY, bandW * 2.f, bandH}, 0.3f,
                            8, 1.2f, outline);
  DrawRectangleRounded(
      {center.x - bandW + 2, bandY + 1, bandW * 2.f - 4, bandH * 0.38f}, 0.3f,
      4, ColorAlpha(light, 0.45f));
  float prongX[5];
  float spread = bandW * 0.92f;
  for (int i = 0; i < 5; i++)
    prongX[i] = center.x - spread + i * (spread * 2.f / 4.f);
  float prongBaseY = bandY + bandH * 0.3f;
  float prongH[5] = {r * 0.72f, r * 0.50f, r * 0.90f, r * 0.50f, r * 0.72f};
  float prongW = r * 0.13f;
  for (int i = 0; i < 5; i++) {
    float tipY = prongBaseY - prongH[i], px = prongX[i];
    DrawRectangleRounded({px - prongW, tipY, prongW * 2.f, prongH[i]}, 0.4f, 6,
                         body);
    DrawRectangleRoundedLines({px - prongW, tipY, prongW * 2.f, prongH[i]},
                              0.4f, 6, 1.1f, outline);
    float orbR = prongW * 1.15f;
    DrawCircleV({px, tipY}, orbR, body);
    DrawCircleLines((int)px, (int)tipY, orbR, outline);
    DrawCircleV({px - orbR * 0.28f, tipY - orbR * 0.28f}, orbR * 0.28f,
                ColorAlpha(light, 0.55f));
  }
  DrawRectangleRounded({center.x - bandW, bandY, bandW * 2.f, bandH}, 0.3f, 8,
                       body);
  DrawRectangleRoundedLines({center.x - bandW, bandY, bandW * 2.f, bandH}, 0.3f,
                            8, 1.2f, outline);
  DrawRectangleRounded(
      {center.x - bandW + 2, bandY + 1, bandW * 2.f - 4, bandH * 0.38f}, 0.3f,
      4, ColorAlpha(light, 0.40f));
  if (isOptimal) {
    DrawRectangleRoundedLines(
        {center.x - bandW - 2, bandY - 2, bandW * 2.f + 4, bandH + 4}, 0.3f, 8,
        1.5f, ColorAlpha(PAL.gold, 0.9f));
    for (int i = 0; i < 5; i++) {
      float orbR = prongW * 1.15f, tipY = prongBaseY - prongH[i];
      DrawCircleLines((int)prongX[i], (int)tipY, orbR + 1.5f,
                      ColorAlpha(PAL.gold, 0.85f));
    }
  }
}

void DrawPanel(Rectangle r, const char *title, Font font,
               Color borderCol = {35, 40, 65, 255}) {
  DrawRectangleRec(r, PAL.panel);
  DrawRectangleRoundedLines(r, 0.04f, 12, 1.0f, borderCol);
  if (title && title[0]) {
    DrawRectangle((int)r.x + 1, (int)r.y + 1, (int)r.width - 2, 30,
                  ColorAlpha(borderCol, 0.3f));
    DrawTextEx(font, title, {r.x + 12, r.y + 8}, 16, 1, PAL.text);
  }
}

void DrawGraphAxes(Rectangle r, Font font, float maxVal, int steps) {
  for (int i = 0; i <= steps; i++) {
    float y = r.y + r.height - (float)i / steps * r.height;
    float val = maxVal * i / steps;
    DrawLineEx({r.x, y}, {r.x + r.width, y}, 1.f, ColorAlpha(PAL.axis, 0.4f));
    DrawTextEx(font, TextFormat("%.0f", val), {r.x - 30, y - 7}, 11, 1,
               PAL.dimText);
  }
  DrawLineEx({r.x, r.y}, {r.x, r.y + r.height}, 1.5f, PAL.axis);
  DrawLineEx({r.x, r.y + r.height}, {r.x + r.width, r.y + r.height}, 1.5f,
             PAL.axis);
}

void DrawSeries(Rectangle bounds, const std::vector<float> &data, int maxIter,
                float maxVal, Color col, float alpha = 1.f, bool fill = false) {
  if (data.size() < 2)
    return;
  float gw = bounds.width, gh = bounds.height;
  float ox = bounds.x, oy = bounds.y + gh;
  if (fill && data.size() > 1) {
    for (int i = 0; i < (int)data.size() - 1; i++) {
      float x1 = ox + (float)i / maxIter * gw,
            x2 = ox + (float)(i + 1) / maxIter * gw;
      float y1 = oy - (data[i] / maxVal) * gh,
            y2 = oy - (data[i + 1] / maxVal) * gh;
      DrawTriangle({x1, oy}, {x1, y1}, {x2, y2}, ColorAlpha(col, 0.07f));
      DrawTriangle({x1, oy}, {x2, y2}, {x2, oy}, ColorAlpha(col, 0.07f));
    }
  }
  for (int i = 0; i < (int)data.size() - 1; i++) {
    Vector2 p1 = {ox + (float)i / maxIter * gw, oy - (data[i] / maxVal) * gh};
    Vector2 p2 = {ox + (float)(i + 1) / maxIter * gw,
                  oy - (data[i + 1] / maxVal) * gh};
    DrawLineEx(p1, p2, 2.f, ColorAlpha(col, alpha));
  }
  if (!data.empty()) {
    float ex = ox + (float)(data.size() - 1) / maxIter * gw;
    float ey = oy - (data.back() / maxVal) * gh;
    DrawCircleV({ex, ey}, 7.f, col);
    DrawCircleV({ex, ey}, 3.f, WHITE);
  }
}

void DrawHeatMap(Rectangle bounds, const RunStats &stats, Font font,
                 int totalSamples) {
  float cw = bounds.width / N, ch = bounds.height / N;
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
      if (t > 0.5f)
        DrawTextEx(font, TextFormat("%d%%", (int)(t * 100)),
                   {cell.x + 2, cell.y + 2}, 9, 1, ColorAlpha(WHITE, 0.7f));
    }
  }
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
// SECTION 9: BUTTON + SLIDER
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

struct Slider {
  Rectangle track;
  float value = 0.5f;
  float handleT = 0.f;
  bool dragging = false;

  void update() {
    Vector2 mp = GetMousePosition();
    float hx = track.x + value * track.width;
    bool hoverHandle =
        CheckCollisionPointCircle(mp, {hx, track.y + track.height / 2}, 14.f);
    Rectangle clickArea = {track.x, track.y - 12.f, track.width,
                           track.height + 24.f};
    bool hoverTrack = CheckCollisionPointRec(mp, clickArea);
    handleT = Lerp(handleT, hoverHandle || dragging ? 1.f : 0.f,
                   GetFrameTime() * 10.f);
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && (hoverHandle || hoverTrack))
      dragging = true;
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON))
      dragging = false;
    if (dragging)
      value = Clamp((mp.x - track.x) / track.width, 0.f, 1.f);
  }
  void draw(Font font, Color col) {
    DrawRectangleRounded(track, 1.f, 8, ColorAlpha(PAL.axis, 0.5f));
    Rectangle fill = track;
    fill.width = value * track.width;
    DrawRectangleRounded(fill, 1.f, 8, ColorAlpha(col, 0.6f));
    float hx = track.x + value * track.width, hy = track.y + track.height / 2;
    DrawCircleV({hx, hy}, 10.f + handleT * 3.f, col);
    DrawCircleV({hx, hy}, 5.f, WHITE);
    DrawTextEx(font, TextFormat("Speed: %.1fx", value * 5.f + 0.2f),
               {track.x + track.width + 12, hy - 8}, 14, 1, PAL.text);
  }
  float getInterval() { return Lerp(1.0f, 0.01f, value); }
};

// ============================================================
// SECTION 10: SWARM PANEL DRAW
// ============================================================
void DrawSwarmPanel(FAEngine &fa, float time, bool showTrails,
                    ParticleSystem &ps) {
  int bestIdx = 0;
  for (int i = 1; i < (int)fa.pop.size(); i++)
    if (fa.pop[i].fitness > fa.pop[bestIdx].fitness)
      bestIdx = i;
  if (fa.pop[bestIdx].screenPos.x > 0) {
    float pulseR = (12.f + fa.pop[bestIdx].fitness / MAX_FIT * 8.f) *
                   (1.4f + 0.4f * sinf(time * 5.f));
    DrawCircleLines((int)fa.pop[bestIdx].screenPos.x,
                    (int)fa.pop[bestIdx].screenPos.y, pulseR,
                    ColorAlpha(PAL.gold, 0.5f));
    DrawCircleLines((int)fa.pop[bestIdx].screenPos.x,
                    (int)fa.pop[bestIdx].screenPos.y, pulseR * 0.7f,
                    ColorAlpha(PAL.gold, 0.3f));
  }
  static int trailFrame = 0;
  trailFrame++;
  for (int idx = 0; idx < (int)fa.pop.size(); idx++) {
    auto &ff = fa.pop[idx];
    if (showTrails && (trailFrame % 11 == idx % 11) &&
        Vector2Distance(ff.screenPos, ff.targetPos) > 3.f)
      ps.emitTrail(ff.screenPos, ColorAlpha(ff.tint, 0.6f));
    for (int t = 1; t < (int)ff.trail.pts.size(); t++) {
      float a = (float)t / ff.trail.pts.size();
      DrawLineEx(ff.trail.pts[t - 1], ff.trail.pts[t], 1.5f * a,
                 ColorAlpha(ff.tint, a * 0.4f));
    }
    float bestDist = 9999.f;
    Vector2 bestNeighbour = {};
    for (auto &other : fa.pop)
      if (&other != &ff && other.fitness > ff.fitness) {
        float d = Vector2Distance(ff.screenPos, other.screenPos);
        if (d < bestDist) {
          bestDist = d;
          bestNeighbour = other.screenPos;
        }
      }
    if (bestDist < 120.f)
      DrawLineEx(ff.screenPos, bestNeighbour, 1.f,
                 ColorAlpha(ff.tint, 0.1f + 0.1f * (1.f - bestDist / 120.f)));
    float fitRatio = std::min(1.f, ff.fitness / MAX_FIT);
    auto stepFit = [](float r) -> float {
      int pct = (int)(r * 100.f);
      if (pct < 3)
        return 0.f;
      if (pct < 5)
        return 0.03f;
      int s = (pct % 2 == 0) ? pct - 1 : pct;
      return (float)s / 100.f;
    };
    float sr = stepFit(fitRatio);
    Color darkStart =
        fa.isModified ? Color{70, 10, 10, 255} : Color{10, 15, 70, 255};
    Color midColor =
        fa.isModified ? Color{180, 50, 20, 255} : Color{25, 75, 200, 255};
    Color ffColor;
    if (sr < 0.5f)
      ffColor = LerpColor(darkStart, midColor, sr * 2.f);
    else if (sr < 0.92f)
      ffColor = LerpColor(midColor, ff.tint, (sr - 0.5f) / 0.42f);
    else
      ffColor = LerpColor(ff.tint, PAL.gold, (sr - 0.92f) / 0.08f * 0.45f);
    if (fitRatio > 0.95f)
      ffColor = LerpColor(ffColor, PAL.gold, (fitRatio - 0.95f) * 5.f * 0.4f);
    float r = 3.f + fitRatio * 6.f;
    bool moving = Vector2Distance(ff.screenPos, ff.targetPos) > 5.f;
    DrawFireflyDetailed(ff.screenPos, r, ffColor, ff.pulsePhase, ff.isElite,
                        time, moving);
  }
  DrawRectangle((int)(SWARM_RECT.x + 4),
                (int)(SWARM_RECT.y + SWARM_RECT.height - 8),
                (int)((SWARM_RECT.width - 8) * fa.diversity), 5,
                ColorAlpha(fa.isModified ? PAL.mod : PAL.orig, 0.7f));
}

// ============================================================
// SECTION 11: COMPARISON RESULT COMPUTATION & DRAW
// ============================================================
static ComparisonResult ComputeComparison(const FAEngine &origFA,
                                          const FAEngine &modFA,
                                          const ModParams &mp) {
  ComparisonResult r;
  r.computed = true;
  r.origBest = origFA.stats.bestFit;
  r.modBest = modFA.stats.bestFit;
  r.origIterOpt = origFA.stats.iterToOptimal;
  r.modIterOpt = modFA.stats.iterToOptimal;
  r.origOptHits = origFA.stats.optimalCount;
  r.modOptHits = modFA.stats.optimalCount;
  r.origAvg = origFA.stats.avgFit;
  r.modAvg = modFA.stats.avgFit;
  r.deltaFit = r.modBest - r.origBest;
  r.deltaAvg = r.modAvg - r.origAvg;
  r.deltaHits = r.modOptHits - r.origOptHits;
  // For iter: both found solution
  if (r.origIterOpt >= 0 && r.modIterOpt >= 0)
    r.deltaIter = r.origIterOpt - r.modIterOpt; // positive = mod was faster
  else if (r.origIterOpt < 0 && r.modIterOpt >= 0)
    r.deltaIter = 999; // only mod found
  else if (r.origIterOpt >= 0 && r.modIterOpt < 0)
    r.deltaIter = -999; // only orig found
  else
    r.deltaIter = 0;

  r.fitImproved = r.deltaFit > 0.f;
  r.iterImproved = r.deltaIter > 0;
  r.hitsImproved = r.deltaHits > 0;
  r.avgImproved = r.deltaAvg > 0.f;
  r.score = (r.fitImproved ? 1 : 0) + (r.iterImproved ? 1 : 0) +
            (r.hitsImproved ? 1 : 0) + (r.avgImproved ? 1 : 0);
  r.usedParams = mp;
  return r;
}

// Draw the post-run comparison inside the METRICS_RECT
static void DrawComparisonPanel(Rectangle bounds, const ComparisonResult &cmp,
                                Font font, float time) {
  // Title
  DrawRectangle((int)bounds.x + 1, (int)bounds.y + 1, (int)bounds.width - 2, 30,
                ColorAlpha(PAL.panelBord, 0.3f));
  DrawTextEx(font, "PARAMETER IMPACT ANALYSIS", {bounds.x + 12, bounds.y + 8},
             14, 1, PAL.gold);

  float y = bounds.y + 38.f;
  float xL = bounds.x + 12.f;
  float xM = bounds.x + bounds.width * 0.38f;
  float xR = bounds.x + bounds.width * 0.62f;

  // Column headers
  DrawTextEx(font, "Metric", {xL, y}, 12, 1, PAL.dimText);
  DrawTextEx(font, "Original", {xM, y}, 12, 1, PAL.orig);
  DrawTextEx(font, "Custom Mod", {xR, y}, 12, 1, PAL.mod);
  y += 18.f;
  DrawLineEx({bounds.x + 6, y}, {bounds.x + bounds.width - 6, y}, 0.5f,
             PAL.panelBord);
  y += 4.f;

  // Helper: draw one row with delta indicator
  auto drawRow = [&](const char *label, std::string vO, std::string vM,
                     bool improved, bool worsened, bool na) {
    DrawRectangleRec({bounds.x + 4, y - 1, bounds.width - 8, 20},
                     ColorAlpha(PAL.panelBord, 0.2f));
    DrawTextEx(font, label, {xL, y + 2}, 12, 1, PAL.text);
    DrawTextEx(font, vO.c_str(), {xM, y + 2}, 12, 1, PAL.orig);
    Color valCol = na ? PAL.dimText
                      : (improved ? PAL.improved
                                  : (worsened ? PAL.worsened : PAL.neutral));
    DrawTextEx(font, vM.c_str(), {xR, y + 2}, 12, 1, valCol);
    // Delta arrow
    if (!na) {
      float ax = bounds.x + bounds.width - 22.f;
      if (improved) {
        DrawTriangle({ax, y + 14}, {ax + 8, y + 14}, {ax + 4, y + 5},
                     ColorAlpha(PAL.improved, 0.9f));
      } else if (worsened) {
        DrawTriangle({ax, y + 5}, {ax + 8, y + 5}, {ax + 4, y + 14},
                     ColorAlpha(PAL.worsened, 0.9f));
      } else {
        DrawLineEx({ax, y + 9}, {ax + 8, y + 9}, 1.5f,
                   ColorAlpha(PAL.neutral, 0.7f));
      }
    }
    y += 22.f;
  };

  // Best fitness
  drawRow("Best Fitness", TextFormat("%.0f / %d", cmp.origBest, MAX_FIT),
          TextFormat("%.0f / %d", cmp.modBest, MAX_FIT), cmp.fitImproved,
          cmp.deltaFit < 0.f, false);

  // Avg fitness
  drawRow("Avg Fitness", TextFormat("%.1f", cmp.origAvg),
          TextFormat("%.1f", cmp.modAvg), cmp.avgImproved, cmp.deltaAvg < 0.f,
          false);

  // Iter to optimal
  {
    std::string vO = cmp.origIterOpt < 0
                         ? "Not found"
                         : TextFormat("iter %d", cmp.origIterOpt);
    std::string vM = cmp.modIterOpt < 0 ? "Not found"
                                        : TextFormat("iter %d", cmp.modIterOpt);
    bool imp = cmp.deltaIter > 0 && cmp.modIterOpt >= 0;
    bool wor =
        cmp.deltaIter < 0 || (cmp.modIterOpt < 0 && cmp.origIterOpt >= 0);
    drawRow("Iter to Opt", vO, vM, imp, wor,
            (cmp.origIterOpt < 0 && cmp.modIterOpt < 0));
  }

  // Optimal hits
  drawRow("Optimal Hits", TextFormat("%d", cmp.origOptHits),
          TextFormat("%d", cmp.modOptHits), cmp.hitsImproved, cmp.deltaHits < 0,
          false);

  y += 4.f;
  DrawLineEx({bounds.x + 6, y}, {bounds.x + bounds.width - 6, y}, 0.5f,
             PAL.panelBord);
  y += 8.f;

  // Overall verdict
  const char *verdicts[] = {"No improvement", "Slight improvement",
                            "Moderate improvement", "Strong improvement",
                            "Full improvement"};
  Color verdictCols[] = {PAL.worsened, PAL.neutral, PAL.mod, PAL.safe,
                         PAL.gold};
  int vi = Clamp(cmp.score, 0, 4);
  DrawTextEx(font, "Overall:", {xL, y}, 12, 1, PAL.dimText);
  DrawTextEx(font, verdicts[vi], {xL + 58, y}, 13, 1.2f, verdictCols[vi]);
  // Score stars
  for (int s = 0; s < 4; s++) {
    Color sc2 = s < cmp.score ? PAL.gold : ColorAlpha(PAL.panelBord, 0.8f);
    DrawCircleV({bounds.x + bounds.width - 80.f + s * 16, y + 7}, 5.f, sc2);
  }
  y += 22.f;

  // Params used section
  DrawLineEx({bounds.x + 6, y}, {bounds.x + bounds.width - 6, y}, 0.5f,
             PAL.panelBord);
  y += 6.f;
  DrawTextEx(font, "Custom params used:", {xL, y}, 11, 1, PAL.dimText);
  y += 16.f;
  DrawTextEx(font, TextFormat("Alpha0: %.2f", cmp.usedParams.alpha0), {xL, y},
             11, 1, PAL.mod);
  DrawTextEx(font,
             TextFormat("MutRate: %.0f%%", cmp.usedParams.mutationRate * 100),
             {xL + 100, y}, 11, 1, PAL.mod);
  y += 14.f;
  DrawTextEx(font, TextFormat("Elites: %d", cmp.usedParams.eliteCount), {xL, y},
             11, 1, PAL.mod);
  DrawTextEx(font,
             TextFormat("HeurInit: %s", ModParams::heuristicLabel(
                                            cmp.usedParams.heuristicRatio)),
             {xL + 100, y}, 11, 1, PAL.mod);
}

// ============================================================
// SECTION 12: PARAMETER TUNING POPUP
// ============================================================
struct ParamPopup {
  bool visible = false;
  float alpha = 0.f; // fade-in/out
  bool confirmed = false;
  bool cancelled = false;

  // Internal sliders (normalised 0..1, mapped to real ranges in getParams)
  float sAlpha0 = 0.50f;    // maps 0..1 → alpha0 0.1..2.0
  float sMutRate = 0.10f;   // maps 0..1 → mutRate 0..1
  float sHeurRatio = 1.00f; // maps 0..1 → heurRatio 0..1
  float sElite = 0.33f;     // maps 0..1 → 0,1,2,3,4  (snapped)

  bool dragging[4] = {false, false, false, false};

  float mapAlpha0() const { return 0.1f + sAlpha0 * 1.9f; }
  float mapMutRate() const { return sMutRate; }
  float mapHeuratio() const { return sHeurRatio; }
  int mapElite() const { return (int)roundf(sElite * 4.f); }

  ModParams getParams() const {
    ModParams p;
    p.alpha0 = mapAlpha0();
    p.mutationRate = mapMutRate();
    p.heuristicRatio = mapHeuratio();
    p.eliteCount = mapElite();
    return p;
  }

  // Reset to Modified FA defaults
  void resetDefaults() {
    sAlpha0 = (0.9f - 0.1f) / 1.9f; // alpha0 = 0.9
    sMutRate = 0.20f;
    sHeurRatio = 1.00f;
    sElite = 2.f / 4.f; // eliteCount = 2
  }

  void open() {
    visible = true;
    confirmed = false;
    cancelled = false;
  }
  void close() { visible = false; }

  // Returns true if any slider was interacted with
  bool updateSlider(int idx, float &sv, Rectangle track, Vector2 mp,
                    bool blockOthers) {
    float hx = track.x + sv * track.width;
    bool hoverH =
        CheckCollisionPointCircle(mp, {hx, track.y + track.height / 2}, 12.f);
    Rectangle ca = {track.x, track.y - 12.f, track.width, track.height + 24.f};
    bool hoverT = CheckCollisionPointRec(mp, ca);
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && (hoverH || hoverT) &&
        !blockOthers)
      dragging[idx] = true;
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON))
      dragging[idx] = false;
    if (dragging[idx])
      sv = Clamp((mp.x - track.x) / track.width, 0.f, 1.f);
    return dragging[idx];
  }

  void update(float dt, Font font) {
    alpha = Lerp(alpha, visible ? 1.f : 0.f, dt * 14.f);
    if (!visible && alpha < 0.01f) {
      confirmed = false;
      cancelled = false;
      return;
    }

    Vector2 mp = GetMousePosition();

    float mw = 560.f, mh = 520.f;
    float mx = WIN_W / 2.f - mw / 2.f;
    float my = WIN_H / 2.f - mh / 2.f;

    // Track widths
    float tw = mw - 120.f;
    float tx = mx + 60.f;
    float rowH = 90.f;
    float startY = my + 90.f;

    // Slider positions
    Rectangle tracks[4];
    for (int i = 0; i < 4; i++)
      tracks[i] = {tx, startY + i * rowH + 40.f, tw, 8.f};

    bool anyDragging = dragging[0] || dragging[1] || dragging[2] || dragging[3];
    updateSlider(0, sAlpha0, tracks[0], mp, anyDragging && !dragging[0]);
    updateSlider(1, sMutRate, tracks[1], mp, anyDragging && !dragging[1]);
    updateSlider(2, sHeurRatio, tracks[2], mp, anyDragging && !dragging[2]);
    updateSlider(3, sElite, tracks[3], mp, anyDragging && !dragging[3]);
    // Snap elite to integer
    sElite = roundf(sElite * 4.f) / 4.f;

    if (!visible)
      return;

    // Buttons — Confirm / Cancel / Reset
    float btnY = my + mh - 52.f;
    Rectangle btnConfirm = {mx + mw / 2 - 170, btnY, 140, 38};
    Rectangle btnCancel = {mx + mw / 2 + 10, btnY, 100, 38};
    Rectangle btnReset = {mx + 16, btnY, 90, 38};

    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
      if (CheckCollisionPointRec(mp, btnConfirm)) {
        confirmed = true;
        close();
      }
      if (CheckCollisionPointRec(mp, btnCancel)) {
        cancelled = true;
        close();
      }
      if (CheckCollisionPointRec(mp, btnReset))
        resetDefaults();
    }
    // Close on Escape
    if (IsKeyPressed(KEY_ESCAPE) && visible) {
      cancelled = true;
      close();
    }
  }

  void draw(Font font, float time) {
    if (alpha < 0.01f)
      return;
    float a = alpha;

    float mw = 560.f, mh = 520.f;
    float mx = WIN_W / 2.f - mw / 2.f;
    float my = WIN_H / 2.f - mh / 2.f;

    // Overlay
    DrawRectangle(0, 0, WIN_W, WIN_H, ColorAlpha({0, 0, 8, 255}, 0.75f * a));

    // Modal card
    DrawRectangleRounded({mx, my, mw, mh}, 0.06f, 16,
                         ColorAlpha({12, 14, 30, 255}, a));
    // Animated border — cycles through mod/gold
    Color borderCol =
        LerpColor(PAL.mod, PAL.gold, 0.5f + 0.5f * sinf(time * 2.f));
    DrawRectangleRoundedLines({mx, my, mw, mh}, 0.06f, 16, 2.f,
                              ColorAlpha(borderCol, a));

    // Header stripe
    DrawRectangleRounded({mx, my, mw, 60.f}, 0.06f, 16,
                         ColorAlpha({18, 16, 45, 255}, a));
    DrawRectangle((int)mx + 2, (int)(my + 58), (int)mw - 4, 2,
                  ColorAlpha(PAL.mod, a * 0.5f));
    DrawTextEx(font, "CONFIGURE MODIFIED FA PARAMETERS", {mx + 20, my + 12}, 18,
               1.5f, ColorAlpha(PAL.gold, a));
    DrawTextEx(font, "Drag sliders — changes apply to next run",
               {mx + 20, my + 36}, 12, 1, ColorAlpha(PAL.dimText, a));

    float tw = mw - 120.f;
    float tx = mx + 60.f;
    float rowH = 90.f;
    float startY = my + 78.f;

    // ── PARAMETER ROWS ──
    struct ParamDef {
      const char *name;
      const char *desc;
      const char *minLabel;
      const char *maxLabel;
      Color col;
    };
    ParamDef defs[4] = {
        {"α₀  Initial Randomness",
         "Controls early exploration breadth. Higher = wider initial search.",
         "0.1 (tight)", "2.0 (wild)", PAL.gold},
        {"Mutation Rate",
         "Swap probability per firefly. Higher = more diversity injection.",
         "0% (none)",
         "100% (always)",
         {200, 100, 255, 255}},
        {"Heuristic Init Ratio",
         "Fraction of population using greedy placement vs random start.",
         "Random", "Fully Heuristic", PAL.safe},
        {"Elite Count",
         "Top-N fireflies preserved each iteration. 0 = elitism off.",
         "0 (off)", "4 (heavy)", PAL.mod},
    };

    float sVals[4] = {sAlpha0, sMutRate, sHeurRatio, sElite};
    Color sliderCols[4] = {PAL.gold, {200, 100, 255, 255}, PAL.safe, PAL.mod};

    // Current value labels
    const char *valLabels[4];
    valLabels[0] = TextFormat("%.2f", mapAlpha0());
    valLabels[1] = TextFormat("%.0f%%", mapMutRate() * 100);
    valLabels[2] = ModParams::heuristicLabel(mapHeuratio());
    valLabels[3] = TextFormat("%d fireflies", mapElite());

    for (int i = 0; i < 4; i++) {
      float ry = startY + i * rowH;
      Rectangle track = {tx, ry + 44.f, tw, 8.f};

      // Row background
      DrawRectangleRounded(
          {mx + 8, ry + 4, mw - 16, rowH - 8}, 0.12f, 8,
          ColorAlpha(PAL.panelBord, a * (dragging[i] ? 0.5f : 0.25f)));
      if (dragging[i])
        DrawRectangleRoundedLines({mx + 8, ry + 4, mw - 16, rowH - 8}, 0.12f, 8,
                                  1.f, ColorAlpha(sliderCols[i], a * 0.6f));

      // Param name + value badge
      DrawTextEx(font, defs[i].name, {tx, ry + 10.f}, 14, 1,
                 ColorAlpha(sliderCols[i], a));
      // Value badge
      Vector2 vs = MeasureTextEx(font, valLabels[i], 13, 1);
      float bw2 = vs.x + 18.f;
      float bx = mx + mw - 16.f - bw2;
      DrawRectangleRounded({bx, ry + 8.f, bw2, 22.f}, 0.4f, 8,
                           ColorAlpha(sliderCols[i], a * 0.2f));
      DrawRectangleRoundedLines({bx, ry + 8.f, bw2, 22.f}, 0.4f, 8, 0.8f,
                                ColorAlpha(sliderCols[i], a * 0.7f));
      DrawTextEx(font, valLabels[i], {bx + 9, ry + 11.f}, 13, 1,
                 ColorAlpha(sliderCols[i], a));

      // Description
      DrawTextEx(font, defs[i].desc, {tx, ry + 26.f}, 11, 1,
                 ColorAlpha(PAL.dimText, a * 0.8f));

      // Slider track
      DrawRectangleRounded(track, 1.f, 8, ColorAlpha(PAL.axis, a * 0.5f));
      Rectangle fill = track;
      fill.width = sVals[i] * track.width;
      DrawRectangleRounded(fill, 1.f, 8, ColorAlpha(sliderCols[i], a * 0.6f));

      // Handle
      float hx2 = track.x + sVals[i] * track.width;
      float hy2 = track.y + track.height / 2;
      float hr = dragging[i] ? 13.f : 10.f;
      DrawCircleV({hx2, hy2}, hr, ColorAlpha(sliderCols[i], a));
      DrawCircleV({hx2, hy2}, hr * 0.45f, ColorAlpha(WHITE, a * 0.9f));
      // Glow when dragging
      if (dragging[i])
        DrawCircleV({hx2, hy2}, hr * 1.8f,
                    ColorAlpha(sliderCols[i], a * 0.15f));

      // Min / max labels
      DrawTextEx(font, defs[i].minLabel, {track.x, ry + 56.f}, 10, 1,
                 ColorAlpha(PAL.dimText, a * 0.7f));
      Vector2 maxSz = MeasureTextEx(font, defs[i].maxLabel, 10, 1);
      DrawTextEx(font, defs[i].maxLabel,
                 {track.x + track.width - maxSz.x, ry + 56.f}, 10, 1,
                 ColorAlpha(PAL.dimText, a * 0.7f));
    }

    // ── BUTTONS ──
    float btnY = my + mh - 52.f;
    Rectangle btnConfirm = {mx + mw / 2 - 170, btnY, 140, 38};
    Rectangle btnCancel = {mx + mw / 2 + 10, btnY, 100, 38};
    Rectangle btnReset = {mx + 16, btnY, 90, 38};

    Vector2 mp = GetMousePosition();
    auto drawBtn = [&](Rectangle r, const char *lbl, Color bc) {
      bool hov = CheckCollisionPointRec(mp, r);
      DrawRectangleRounded(
          r, 0.35f, 12,
          ColorAlpha(hov ? ColorBrightness(bc, 0.2f) : bc, a * 0.85f));
      DrawRectangleRoundedLines(
          r, 0.35f, 12, 1.5f,
          ColorAlpha(hov ? WHITE : LerpColor(bc, WHITE, 0.4f), a));
      Vector2 ts = MeasureTextEx(font, lbl, 14, 1);
      DrawTextEx(font, lbl,
                 {r.x + r.width / 2 - ts.x / 2, r.y + r.height / 2 - ts.y / 2},
                 14, 1, ColorAlpha(WHITE, a));
    };
    drawBtn(btnConfirm, "Run Modified FA", PAL.mod);
    drawBtn(btnCancel, "Cancel", PAL.conflict);
    drawBtn(btnReset, "Defaults", ColorBrightness(PAL.panelBord, 0.4f));

    // Keyboard hint
    DrawTextEx(font, "[Esc] to cancel", {mx + mw - 100, my + mh - 18}, 10, 1,
               ColorAlpha(PAL.dimText, a * 0.5f));
  }
};

// ============================================================
// SECTION 13: STATS TICKER
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
    float col1 = bounds.x + 10, col2 = bounds.x + bounds.width * 0.5f;
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
// SECTION 14: STARFIELD
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
// SECTION 15: MAIN
// ============================================================
int main() {
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
  InitWindow(WIN_W, WIN_H,
             "Firefly Algorithm Visualizer — Parameter Tuning Edition");
  SetTargetFPS(60);

  Font fontBold = GetFontDefault();
  Font fontSm = GetFontDefault();

  Shader bloomShader = LoadShaderFromMemory(nullptr, BLOOM_FRAG);
  Shader vignetteShader = LoadShaderFromMemory(nullptr, VIGNETTE_FRAG);
  int bloomResLoc = GetShaderLocation(bloomShader, "resolution");
  int bloomIntLoc = GetShaderLocation(bloomShader, "intensity");
  Vector2 res = {(float)WIN_W, (float)WIN_H};
  SetShaderValue(bloomShader, bloomResLoc, &res, SHADER_UNIFORM_VEC2);
  RenderTexture2D bloomTarget = LoadRenderTexture(WIN_W, WIN_H);

  Starfield starfield;
  starfield.init(300);
  ParticleSystem particles;
  StatsTicker ticker;
  Slider speedSlider;
  speedSlider.value = 0.4f;

  FAEngine origFA, modFA;
  BoardAnimator animO, animM;

  // Parameter popup
  ParamPopup paramPopup;
  paramPopup.resetDefaults();
  ModParams currentModParams = paramPopup.getParams();

  // Comparison result
  ComparisonResult cmpResult;
  bool showComparison = false;

  // Buttons
  Button btnOrig = {{0, 0, 118, 44}, "Original FA", PAL.orig};
  Button btnMod = {{0, 0, 118, 44}, "Custom Mod FA", PAL.mod};
  Button btnBoth = {{0, 0, 100, 44}, "Run Both", PAL.accent};
  Button btnReset = {{0, 0, 80, 44}, "Reset", PAL.conflict};
  Button btnPlayStop = {{0, 0, 108, 40}, "PLAY", PAL.safe};
  Button btnStepFwd = {{0, 0, 52, 40}, ">|", PAL.gold};
  Button btnStepBwd = {{0, 0, 52, 40}, "|<", ColorBrightness(PAL.gold, -0.2f)};
  Button btnParams = {{0, 0, 90, 40}, "Params", {120, 60, 200, 255}};
  Button btnExpandGraph = {{0, 0, 84, 24}, "Expand", PAL.accent};
  Button btnCloseGraph = {
      {WIN_W / 2.f + 500.f - 85, WIN_H / 2.f - 300.f + 4, 75, 22},
      "Close",
      PAL.conflict};
  Button btnTheory = {{0, 0, 86, 44}, "Theory", {80, 60, 160, 255}};
  bool showTheory = false;
  int theoryTab = 0;
  float theoryAlpha = 0.f;

  // Step history
  struct SnapShot {
    std::vector<std::vector<int>> posO, posM;
    std::vector<float> fitO, fitM;
    int iterO = 0, iterM = 0;
    bool doneO = false, doneM = false;
  };
  std::deque<SnapShot> stepHistory;
  static const int MAX_HISTORY = 30;

  bool showO = false, showM = false, running = false;
  float stepTimer = 0.f, time = 0.f;
  bool showConflicts = true, showTrails = true, showCmpTable = false;
  float celebTimer = 0.f;
  bool showExpandedGraph = false, hasAutoExpanded = false;
  bool solutionFound = false, everSolved = false;
  float solvedBadgeX = (float)WIN_W + 10.f;
  float expandedT = 0.f;
  int boardQueenFilter = 0;
  bool showDownload = false;
  std::string downloadMsg = "";
  float downloadMsgTimer = 0.f;

  auto resetRun = [&]() {
    showO = showM = running = showCmpTable = false;
    showExpandedGraph = false;
    hasAutoExpanded = false;
    solutionFound = false;
    everSolved = false;
    solvedBadgeX = (float)WIN_W + 10.f;
    showDownload = false;
    downloadMsgTimer = 0.f;
    stepHistory.clear();
    btnPlayStop.label = "PLAY";
    btnPlayStop.col = PAL.safe;
    cmpResult = ComparisonResult{};
    showComparison = false;
  };

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();
    time += dt;
    stepTimer += dt;
    expandedT = Lerp(expandedT, showExpandedGraph ? 1.f : 0.f, dt * 10.f);
    if (downloadMsgTimer > 0.f)
      downloadMsgTimer -= dt;

    // Update popup (must be before buttons to allow its buttons to fire first)
    bool popupBlocksInput = paramPopup.visible;
    paramPopup.update(dt, fontSm);
    if (paramPopup.confirmed) {
      currentModParams = paramPopup.getParams();
      // Auto-start Modified FA with new params
      recalcParams();
      modFA.swarmHalf = showO ? 2 : 0;
      modFA.init(true, PAL.mod, &currentModParams);
      animM.init(N, BOARD_RECT);
      showM = true;
      running = true;
      cmpResult = ComparisonResult{};
      showComparison = false;
      stepTimer = 0.f;
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
      stepHistory.clear();
    }

    // Only update buttons when popup not blocking
    if (!popupBlocksInput) {
      btnOrig.update(dt);
      btnMod.update(dt);
      btnBoth.update(dt);
      btnReset.update(dt);
      btnPlayStop.update(dt);
      btnStepFwd.update(dt);
      btnStepBwd.update(dt);
      btnParams.update(dt);
      btnTheory.update(dt);
      if (btnTheory.pressed)
        showTheory = !showTheory;
      speedSlider.update();
      btnExpandGraph.update(dt);
      if (expandedT > 0.01f)
        btnCloseGraph.update(dt);
    }
    theoryAlpha = Lerp(theoryAlpha, showTheory ? 1.f : 0.f, dt * 12.f);

    if (!popupBlocksInput) {
      // Params button opens popup
      if (btnParams.pressed)
        paramPopup.open();

      // Expand graph
      if (btnExpandGraph.pressed)
        showExpandedGraph = true;
      if (btnCloseGraph.pressed)
        showExpandedGraph = false;

      // Original FA
      if (btnOrig.pressed) {
        recalcParams();
        origFA.swarmHalf = 0;
        origFA.init(false, PAL.orig);
        animO.init(N, BOARD_RECT);
        showO = true;
        showM = false;
        running = true;
        showCmpTable = false;
        stepTimer = 0.f;
        showExpandedGraph = false;
        hasAutoExpanded = false;
        solutionFound = false;
        stepHistory.clear();
        cmpResult = ComparisonResult{};
        showComparison = false;
        btnPlayStop.label = "STOP";
        btnPlayStop.col = PAL.conflict;
      }
      // Custom Modified FA — opens popup first
      if (btnMod.pressed)
        paramPopup.open();

      // Run Both — opens popup for mod params
      if (btnBoth.pressed)
        paramPopup.open();

      if (btnReset.pressed)
        resetRun();

      if (btnPlayStop.pressed && (showO || showM)) {
        running = !running;
        btnPlayStop.label = running ? "STOP" : "PLAY";
        btnPlayStop.col = running ? PAL.conflict : PAL.safe;
      }
      if (btnStepFwd.pressed && (showO || showM)) {
        SnapShot snap;
        if (showO && origFA.inited) {
          snap.posO.resize(POP);
          snap.fitO.resize(POP);
          for (int i = 0; i < POP; i++) {
            snap.posO[i] = origFA.pop[i].position;
            snap.fitO[i] = origFA.pop[i].fitness;
          }
          snap.iterO = origFA.iter;
          snap.doneO = origFA.done;
        }
        if (showM && modFA.inited) {
          snap.posM.resize(POP);
          snap.fitM.resize(POP);
          for (int i = 0; i < POP; i++) {
            snap.posM[i] = modFA.pop[i].position;
            snap.fitM[i] = modFA.pop[i].fitness;
          }
          snap.iterM = modFA.iter;
          snap.doneM = modFA.done;
        }
        stepHistory.push_back(snap);
        if ((int)stepHistory.size() > MAX_HISTORY)
          stepHistory.pop_front();
        if (showO && !origFA.done)
          origFA.step();
        if (showM && !modFA.done)
          modFA.step();
      }
      if (btnStepBwd.pressed && !stepHistory.empty()) {
        auto &snap = stepHistory.back();
        if (showO && origFA.inited && !snap.posO.empty()) {
          for (int i = 0; i < POP && i < (int)snap.posO.size(); i++) {
            origFA.pop[i].position = snap.posO[i];
            origFA.pop[i].fitness = snap.fitO[i];
          }
          origFA.iter = snap.iterO;
          origFA.done = snap.doneO;
          if ((int)origFA.stats.bestPerIter.size() > origFA.iter)
            origFA.stats.bestPerIter.resize(origFA.iter);
        }
        if (showM && modFA.inited && !snap.posM.empty()) {
          for (int i = 0; i < POP && i < (int)snap.posM.size(); i++) {
            modFA.pop[i].position = snap.posM[i];
            modFA.pop[i].fitness = snap.fitM[i];
          }
          modFA.iter = snap.iterM;
          modFA.done = snap.doneM;
          if ((int)modFA.stats.bestPerIter.size() > modFA.iter)
            modFA.stats.bestPerIter.resize(modFA.iter);
        }
        stepHistory.pop_back();
        running = false;
        btnPlayStop.label = "PLAY";
        btnPlayStop.col = PAL.safe;
      }

      // Keyboard
      if (IsKeyPressed(KEY_O)) {
        recalcParams();
        origFA.swarmHalf = 0;
        origFA.init(false, PAL.orig);
        animO.init(N, BOARD_RECT);
        showO = true;
        showM = false;
        running = true;
        showCmpTable = false;
        stepTimer = 0.f;
        showExpandedGraph = false;
        hasAutoExpanded = false;
        solutionFound = false;
        stepHistory.clear();
        cmpResult = ComparisonResult{};
        showComparison = false;
        btnPlayStop.label = "STOP";
        btnPlayStop.col = PAL.conflict;
      }
      if (IsKeyPressed(KEY_M))
        paramPopup.open();
      if (IsKeyPressed(KEY_B))
        paramPopup.open();
      if (IsKeyPressed(KEY_R))
        resetRun();
      if (IsKeyPressed(KEY_SPACE) && (showO || showM)) {
        running = !running;
        btnPlayStop.label = running ? "STOP" : "PLAY";
        btnPlayStop.col = running ? PAL.conflict : PAL.safe;
      }
      if (IsKeyPressed(KEY_RIGHT) && (showO || showM)) {
        if (showO && !origFA.done)
          origFA.step();
        if (showM && !modFA.done)
          modFA.step();
      }
      if (IsKeyPressed(KEY_UP) && N < 32) {
        N++;
        recalcParams();
        resetRun();
      }
      if (IsKeyPressed(KEY_DOWN) && N > 4) {
        N--;
        recalcParams();
        resetRun();
      }
      if (IsKeyPressed(KEY_E))
        showExpandedGraph = !showExpandedGraph;
      if (IsKeyPressed(KEY_C))
        showConflicts = !showConflicts;
      if (IsKeyPressed(KEY_T))
        showTrails = !showTrails;
      if (IsKeyPressed(KEY_X))
        showCmpTable = !showCmpTable;
      if (IsKeyPressed(KEY_H))
        showTheory = !showTheory;
    }

    // Simulation step
    if (running && stepTimer >= speedSlider.getInterval()) {
      if (showO && !origFA.done)
        origFA.step();
      if (showM && !modFA.done)
        modFA.step();
      stepTimer = 0.f;

      bool odone = !showO || origFA.done;
      bool mdone = !showM || modFA.done;
      if (odone && mdone && (showO || showM)) {
        showDownload = true;
        if (running) {
          running = false;
          btnPlayStop.label = "PLAY";
          btnPlayStop.col = PAL.safe;
          bool found = (showO && origFA.stats.bestFit >= (float)MAX_FIT) ||
                       (showM && modFA.stats.bestFit >= (float)MAX_FIT);
          if (found) {
            celebTimer = 3.f;
            solutionFound = true;
            everSolved = true;
            particles.emit({WIN_W * 0.5f, WIN_H * 0.4f}, 120, 0, PAL.gold);
          }
          if (!hasAutoExpanded) {
            showExpandedGraph = true;
            hasAutoExpanded = true;
          }
          // Compute comparison when both engines ran
          if (showO && showM && origFA.inited && modFA.inited) {
            cmpResult = ComputeComparison(origFA, modFA, currentModParams);
            showComparison = true;
          }
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

    // Firefly interpolation
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

    if (showO && origFA.inited) {
      animO.setTargets(origFA.getBest().position, N, BOARD_RECT);
      animO.update(dt);
    }
    if (showM && modFA.inited) {
      animM.setTargets(modFA.getBest().position, N, BOARD_RECT);
      animM.update(dt);
    }

    // Ticker
    ticker.clear();
    if (showO) {
      auto b = origFA.getBest();
      ticker.add("Orig Best:", TextFormat("%d/%d", (int)b.fitness, MAX_FIT),
                 PAL.orig);
      ticker.add("Orig Iter:", TextFormat("%d/%d", origFA.iter, MAX_ITER),
                 PAL.dimText);
      if (origFA.stats.iterToOptimal >= 0)
        ticker.add("Orig→Opt:", TextFormat("%d", origFA.stats.iterToOptimal),
                   PAL.safe);
    }
    if (showM) {
      auto b = modFA.getBest();
      ticker.add("Mod Best:", TextFormat("%d/%d", (int)b.fitness, MAX_FIT),
                 PAL.mod);
      ticker.add("Mod Iter:", TextFormat("%d/%d", modFA.iter, MAX_ITER),
                 PAL.dimText);
      if (modFA.stats.iterToOptimal >= 0)
        ticker.add("Mod→Opt:", TextFormat("%d", modFA.stats.iterToOptimal),
                   PAL.safe);
    }

    // ====================================================
    // BLOOM TARGET
    // ====================================================
    BeginTextureMode(bloomTarget);
    ClearBackground(BLANK);
    if (showO && origFA.inited && origFA.stats.bestFit >= (float)MAX_FIT) {
      auto b = origFA.getBest();
      float cell = BOARD_RECT.height / N;
      for (int r = 0; r < N; r++)
        DrawCircleV({BOARD_RECT.x + b.position[r] * cell + cell / 2,
                     BOARD_RECT.y + r * cell + cell / 2},
                    cell * 0.4f, ColorAlpha(PAL.gold, 0.3f));
    }
    if (showM && modFA.inited && modFA.stats.bestFit >= (float)MAX_FIT) {
      auto b = modFA.getBest();
      float cell = BOARD_RECT.height / N;
      for (int r = 0; r < N; r++)
        DrawCircleV({BOARD_RECT.x + b.position[r] * cell + cell / 2,
                     BOARD_RECT.y + r * cell + cell / 2},
                    cell * 0.4f, ColorAlpha(PAL.mod, 0.2f));
    }
    if (showO)
      for (auto &ff : origFA.pop)
        DrawCircleV(ff.screenPos, (8.f + (ff.fitness / MAX_FIT) * 12.f) * 2.f,
                    ColorAlpha(PAL.orig, 0.08f));
    if (showM)
      for (auto &ff : modFA.pop)
        DrawCircleV(ff.screenPos, (8.f + (ff.fitness / MAX_FIT) * 12.f) * 2.f,
                    ColorAlpha(PAL.mod, 0.08f));
    EndTextureMode();

    // ====================================================
    // MAIN DRAW
    // ====================================================
    BeginDrawing();
    ClearBackground(PAL.bg);
    starfield.draw();
    for (int i = 8; i >= 1; i--)
      DrawCircleV({WIN_W * 0.5f, WIN_H * 0.5f}, (float)i * 120.f,
                  ColorAlpha({20, 25, 60, 255}, 0.02f));

    // ── TITLE BAR ──
    DrawRectangleGradientH(0, 0, WIN_W, 56, {8, 10, 28, 255},
                           {18, 12, 40, 255});
    DrawRectangle(0, 0, 4, 56, PAL.gold);
    DrawTextEx(fontBold, "FIREFLY ALGORITHM", {18, 6}, 22, 1.5f, PAL.gold);
    DrawTextEx(fontSm, "N-QUEENS  |  PARAMETER TUNING EDITION", {20, 30}, 12,
               2.f, PAL.dimText);

    const char *statusStr =
        running ? "RUNNING" : (showO || showM ? "PAUSED" : "IDLE");
    Color statusCol =
        running ? PAL.safe : (showO || showM ? PAL.gold : PAL.dimText);
    float pillH = 40.f, pillW = 100.f, pillY = (56.f - pillH) / 2.f;
    float pillX = WIN_W - 16.f - pillW;
    DrawRectangleRounded({pillX, pillY, pillW, pillH}, 0.4f, 8,
                         ColorAlpha(statusCol, 0.12f));
    DrawRectangleRoundedLines({pillX, pillY, pillW, pillH}, 0.4f, 8, 1.2f,
                              statusCol);
    float dotPulse = 0.6f + 0.4f * sinf(time * 4.f);
    DrawCircleV({pillX + 15.f, pillY + pillH / 2.f},
                running ? 5.f * dotPulse : 4.f, statusCol);
    Vector2 stSz = MeasureTextEx(fontSm, statusStr, 13, 1);
    DrawTextEx(fontSm, statusStr,
               {pillX + 28.f, pillY + pillH / 2.f - stSz.y / 2.f}, 13, 1,
               statusCol);

    // Current mod params summary in title bar (if custom params set)
    {
      Color pc = LerpColor(PAL.mod, PAL.gold, 0.3f);
      DrawTextEx(
          fontSm,
          TextFormat(
              "α₀=%.2f  mut=%.0f%%  elite=%d  heur=%s", currentModParams.alpha0,
              currentModParams.mutationRate * 100, currentModParams.eliteCount,
              ModParams::heuristicLabel(currentModParams.heuristicRatio)),
          {WIN_W - 500.f, pillY + 5.f}, 11, 1, ColorAlpha(pc, 0.75f));
    }

    // ── BOARD PANEL ──
    DrawPanel(BOARD_RECT, nullptr, fontSm, PAL.panelBord);
    float cell = BOARD_RECT.height / N;
    float bx = BOARD_RECT.x, by = BOARD_RECT.y;
    for (int c = 0; c < N; c++)
      DrawTextEx(fontSm, TextFormat("%c", 'A' + c),
                 {bx + c * cell + cell / 2 - 5, by - 20}, 14, 1, PAL.dimText);
    for (int r = 0; r < N; r++)
      DrawTextEx(fontSm, TextFormat("%d", N - r),
                 {bx - 18, by + r * cell + cell / 2 - 8}, 14, 1, PAL.dimText);
    for (int r = 0; r < N; r++)
      for (int c = 0; c < N; c++) {
        Color sq = (r + c) % 2 == 0 ? PAL.lightSq : PAL.darkSq;
        DrawRectangle((int)(bx + c * cell), (int)(by + r * cell), (int)cell,
                      (int)cell, sq);
      }
    auto drawAttackHighlights = [&](const std::vector<int> &pos) {
      for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
          if (std::abs(pos[i] - pos[j]) == std::abs(i - j)) {
            DrawRectangle((int)bx, (int)(by + i * cell), (int)(N * cell),
                          (int)cell, ColorAlpha(PAL.conflict, 0.10f));
            DrawRectangle((int)bx, (int)(by + j * cell), (int)(N * cell),
                          (int)cell, ColorAlpha(PAL.conflict, 0.10f));
          }
    };
    if (showO && origFA.inited)
      drawAttackHighlights(origFA.getBest().position);
    else if (showM && modFA.inited)
      drawAttackHighlights(modFA.getBest().position);
    if (showConflicts) {
      auto drawConflictAnimated = [&](FAEngine &fa, BoardAnimator &anim) {
        if (!fa.inited || !anim.inited)
          return;
        auto p = fa.getBest().position;
        for (int i = 0; i < N; i++)
          for (int j = i + 1; j < N; j++)
            if (std::abs(p[i] - p[j]) == std::abs(i - j)) {
              DrawLineEx(anim.queenScreenPos[i], anim.queenScreenPos[j], 2.5f,
                         ColorAlpha(PAL.conflict, 0.7f));
              DrawCircleV(anim.queenScreenPos[i], 4.f,
                          ColorAlpha(PAL.conflict, 0.9f));
              DrawCircleV(anim.queenScreenPos[j], 4.f,
                          ColorAlpha(PAL.conflict, 0.9f));
            }
      };
      if (showO)
        drawConflictAnimated(origFA, animO);
      if (showM && !showO)
        drawConflictAnimated(modFA, animM);
    }
    if (showO)
      animO.drawTrails(PAL.orig);
    if (showM)
      animM.drawTrails(PAL.mod);
    auto drawQueensAnimated = [&](BoardAnimator &anim, FAEngine &fa, Color qcol,
                                  float scale) {
      if (!fa.inited || !anim.inited)
        return;
      bool opt = fa.getBest().fitness >= (float)MAX_FIT;
      for (int r = 0; r < N; r++)
        DrawQueenPiece(anim.queenScreenPos[r], BOARD_RECT.height / N * scale,
                       qcol, opt, time);
    };
    if (showO && !showM) {
      float ro = origFA.inited
                     ? std::min(1.f, origFA.getBest().fitness / (float)MAX_FIT)
                     : 0.f;
      Color qco = ro < 0.5f
                      ? LerpColor({10, 15, 70, 255}, PAL.origDim, ro * 2.f)
                      : LerpColor(PAL.origDim, PAL.orig, (ro - 0.5f) * 2.f);
      drawQueensAnimated(animO, origFA, qco, 0.72f);
    }
    if (showM && !showO) {
      float rm = modFA.inited
                     ? std::min(1.f, modFA.getBest().fitness / (float)MAX_FIT)
                     : 0.f;
      Color qcm = rm < 0.5f ? LerpColor({70, 10, 10, 255}, PAL.modDim, rm * 2.f)
                            : LerpColor(PAL.modDim, PAL.mod, (rm - 0.5f) * 2.f);
      drawQueensAnimated(animM, modFA, qcm, 0.72f);
    }
    if (showO && showM) {
      float yOff = cell * 0.13f, qSize = cell * 0.56f,
            qSizeSingle = cell * 0.72f;
      bool optO = origFA.inited && origFA.getBest().fitness >= (float)MAX_FIT,
           optM = modFA.inited && modFA.getBest().fitness >= (float)MAX_FIT;
      float ratioO =
                origFA.inited
                    ? std::min(1.f, origFA.getBest().fitness / (float)MAX_FIT)
                    : 0.f,
            ratioM =
                modFA.inited
                    ? std::min(1.f, modFA.getBest().fitness / (float)MAX_FIT)
                    : 0.f;
      Color qColO =
          ratioO < 0.5f
              ? LerpColor({10, 15, 70, 255}, PAL.origDim, ratioO * 2.f)
              : LerpColor(PAL.origDim, PAL.orig, (ratioO - 0.5f) * 2.f);
      Color qColM = ratioM < 0.5f
                        ? LerpColor({70, 10, 10, 255}, PAL.modDim, ratioM * 2.f)
                        : LerpColor(PAL.modDim, PAL.mod, (ratioM - 0.5f) * 2.f);
      float vc = qSize * 0.42f * 0.28f, vcSing = qSizeSingle * 0.42f * 0.28f;
      for (int r = 0; r < N; r++) {
        if (boardQueenFilter == 0) {
          Vector2 posO = animO.queenScreenPos[r];
          posO.y -= yOff;
          Vector2 posM = animM.queenScreenPos[r];
          posM.y = posM.y - vc + yOff * 0.05f;
          DrawQueenPiece(posO, qSize, qColO, optO, time);
          DrawQueenPiece(posM, qSize, qColM, optM, time);
        } else if (boardQueenFilter == 1) {
          Vector2 posO = animO.queenScreenPos[r];
          posO.y -= vcSing;
          DrawQueenPiece(posO, qSizeSingle, qColO, optO, time);
        } else {
          Vector2 posM = animM.queenScreenPos[r];
          posM.y -= vcSing;
          DrawQueenPiece(posM, qSizeSingle, qColM, optM, time);
        }
      }
    }
    DrawRectangleLinesEx(BOARD_RECT, 2, PAL.panelBord);
    if (!showM && showO && origFA.inited) {
      auto b = origFA.getBest();
      DrawTextEx(fontSm, TextFormat("Orig: %d/%d", (int)b.fitness, MAX_FIT),
                 {bx, by + N * cell + 8}, 14, 1, PAL.orig);
    }
    if (!showO && showM && modFA.inited) {
      auto b = modFA.getBest();
      const char *txt = TextFormat("Mod: %d/%d", (int)b.fitness, MAX_FIT);
      Vector2 sz = MeasureTextEx(fontSm, txt, 14, 1);
      DrawTextEx(fontSm, txt, {bx + BOARD_RECT.width - sz.x, by + N * cell + 8},
                 14, 1, PAL.mod);
    }
    if (showO && showM) {
      if (origFA.inited) {
        auto b = origFA.getBest();
        DrawTextEx(fontSm, TextFormat("Orig:%d/%d", (int)b.fitness, MAX_FIT),
                   {bx, by + N * cell + 8}, 13, 1, PAL.orig);
      }
      if (modFA.inited) {
        auto b = modFA.getBest();
        const char *txt = TextFormat("Mod:%d/%d", (int)b.fitness, MAX_FIT);
        Vector2 sz = MeasureTextEx(fontSm, txt, 13, 1);
        DrawTextEx(fontSm, txt,
                   {bx + BOARD_RECT.width - sz.x, by + N * cell + 8}, 13, 1,
                   PAL.mod);
      }
    }
    if (!(showO && showM))
      boardQueenFilter = 0;

    // N Controls
    {
      bool dualMode = showO && showM;
      float nby2 = by + N * cell + 28.f;
      if (!dualMode) {
        float nbw = 160.f, nbx = bx + (BOARD_RECT.width / 2.f) - nbw / 2.f;
        bool hovDec =
            CheckCollisionPointRec(GetMousePosition(), {nbx, nby2, 36, 26});
        bool hovInc = CheckCollisionPointRec(GetMousePosition(),
                                             {nbx + 124, nby2, 36, 26});
        DrawRectangleRounded(
            {nbx, nby2, 36, 26}, 0.4f, 8,
            ColorAlpha(hovDec ? PAL.conflict
                              : ColorBrightness(PAL.conflict, -0.3f),
                       0.9f));
        DrawTextEx(fontBold, "-", {nbx + 12, nby2 + 4}, 16, 1, WHITE);
        DrawRectangleRounded({nbx + 40, nby2, 80, 26}, 0.4f, 8,
                             ColorAlpha(PAL.bg, 0.88f));
        DrawRectangleRoundedLines({nbx + 40, nby2, 80, 26}, 0.4f, 8, 1.2f,
                                  PAL.gold);
        Vector2 nSz = MeasureTextEx(fontBold, TextFormat("N = %d", N), 13, 1);
        DrawTextEx(fontBold, TextFormat("N = %d", N),
                   {nbx + 40 + 40 - nSz.x / 2, nby2 + 6}, 13, 1, PAL.gold);
        DrawRectangleRounded(
            {nbx + 124, nby2, 36, 26}, 0.4f, 8,
            ColorAlpha(hovInc ? PAL.safe : ColorBrightness(PAL.safe, -0.3f),
                       0.9f));
        DrawTextEx(fontBold, "+", {nbx + 136, nby2 + 4}, 16, 1, WHITE);
        if (!popupBlocksInput) {
          if (hovDec && IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && N > 4) {
            N--;
            recalcParams();
            resetRun();
          }
          if (hovInc && IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && N < 32) {
            N++;
            recalcParams();
            resetRun();
          }
        }
      } else {
        const char *bqLabels[] = {"Both", "Blue", "Org"};
        Color bqCols[] = {PAL.accent, PAL.orig, PAL.mod};
        float bqW = 64.f, bqH = 26.f, bqGap = 8.f, totalW = 3 * bqW + 2 * bqGap,
              bqX = bx + (BOARD_RECT.width / 2.f) - totalW / 2.f;
        for (int v = 0; v < 3; v++) {
          float vx = bqX + v * (bqW + bqGap);
          bool sel = boardQueenFilter == v;
          bool hov =
              CheckCollisionPointRec(GetMousePosition(), {vx, nby2, bqW, bqH});
          DrawRectangleRounded(
              {vx, nby2, bqW, bqH}, 0.4f, 8,
              ColorAlpha(sel ? bqCols[v] : ColorBrightness(bqCols[v], -0.35f),
                         sel ? 1.f : 0.6f));
          if (sel)
            DrawRectangleRoundedLines({vx, nby2, bqW, bqH}, 0.4f, 8, 1.5f,
                                      WHITE);
          Vector2 bsz = MeasureTextEx(fontBold, bqLabels[v], 13, 1);
          DrawTextEx(fontBold, bqLabels[v],
                     {vx + bqW / 2 - bsz.x / 2, nby2 + bqH / 2 - bsz.y / 2}, 13,
                     1, WHITE);
          if (!popupBlocksInput && hov &&
              IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            boardQueenFilter = v;
        }
      }
    }

    // ── SWARM PANEL ──
    {
      static int swarmView = 0;
      bool bothRunning = showO && showM;
      if (bothRunning) {
        if (swarmView == 1) {
          origFA.swarmHalf = 0;
          origFA.layoutSwarm(0);
          modFA.swarmHalf = 2;
          modFA.layoutSwarm(2);
        } else if (swarmView == 2) {
          modFA.swarmHalf = 0;
          modFA.layoutSwarm(0);
          origFA.swarmHalf = 1;
          origFA.layoutSwarm(1);
        } else {
          origFA.swarmHalf = 1;
          origFA.layoutSwarm(1);
          modFA.swarmHalf = 2;
          modFA.layoutSwarm(2);
        }
      }
      const char *panelTitle =
          (!bothRunning) ? (showM && !showO ? "Custom Modified FA Swarm"
                                            : "Original FA Swarm")
                         : (swarmView == 1   ? "Original FA Swarm"
                            : swarmView == 2 ? "Custom Modified FA Swarm"
                                             : "Dual Swarm");
      Color swBorder = running ? LerpColor(PAL.panelBord, PAL.safe,
                                           0.5f + 0.5f * sinf(time * 4.f))
                               : PAL.panelBord;
      DrawPanel(SWARM_RECT, panelTitle, fontSm, swBorder);
      if (bothRunning) {
        const char *viewLabels[] = {"Dual", "Orig", "Mod"};
        float tvW = 48.f, tvH = 18.f,
              tvX = SWARM_RECT.x + SWARM_RECT.width - tvW * 3 - 10,
              tvY = SWARM_RECT.y + 7;
        for (int v = 0; v < 3; v++) {
          float vx = tvX + v * tvW;
          bool sel = swarmView == v;
          bool hov = !popupBlocksInput &&
                     CheckCollisionPointRec(GetMousePosition(),
                                            {vx, tvY, tvW - 3, tvH});
          Color vc2 = v == 0 ? PAL.accent : (v == 1 ? PAL.orig : PAL.mod);
          DrawRectangleRounded(
              {vx, tvY, tvW - 3, tvH}, 0.4f, 8,
              ColorAlpha(sel ? vc2 : ColorBrightness(vc2, -0.35f),
                         sel ? 1.f : 0.55f));
          if (sel)
            DrawRectangleRoundedLines({vx, tvY, tvW - 3, tvH}, 0.4f, 8, 1.5f,
                                      WHITE);
          Vector2 vsz = MeasureTextEx(fontSm, viewLabels[v], 11, 1);
          DrawTextEx(
              fontSm, viewLabels[v],
              {vx + (tvW - 3) / 2 - vsz.x / 2, tvY + tvH / 2 - vsz.y / 2}, 11,
              1, WHITE);
          if (hov && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            swarmView = v;
        }
      }
      BeginScissorMode((int)SWARM_RECT.x, (int)SWARM_RECT.y,
                       (int)SWARM_RECT.width, (int)SWARM_RECT.height);
      if (!bothRunning) {
        if (showO)
          DrawSwarmPanel(origFA, time, showTrails, particles);
        if (showM)
          DrawSwarmPanel(modFA, time, showTrails, particles);
      } else if (swarmView == 1)
        DrawSwarmPanel(origFA, time, showTrails, particles);
      else if (swarmView == 2)
        DrawSwarmPanel(modFA, time, showTrails, particles);
      else {
        DrawSwarmPanel(origFA, time, showTrails, particles);
        DrawSwarmPanel(modFA, time, showTrails, particles);
      }
      EndScissorMode();
      // Progress bar
      float progO =
                (showO && origFA.inited) ? (float)origFA.iter / MAX_ITER : 0.f,
            progM =
                (showM && modFA.inited) ? (float)modFA.iter / MAX_ITER : 0.f;
      int active = (showO ? 1 : 0) + (showM ? 1 : 0);
      float combined = active > 0 ? (progO + progM) / active : 0.f;
      float barX = SWARM_RECT.x + 10,
            barY = SWARM_RECT.y + SWARM_RECT.height - 24,
            barW = SWARM_RECT.width - 20, barH = 14.f;
      DrawRectangleRounded({barX, barY, barW, barH}, 1.f, 8,
                           ColorAlpha(PAL.panelBord, 0.5f));
      if (combined > 0.f) {
        float fillW = combined * barW;
        Color cLeft = showO ? PAL.orig : PAL.mod,
              cRight = showM ? PAL.mod : PAL.orig;
        DrawRectangleGradientH((int)barX, (int)barY, (int)fillW, (int)barH,
                               cLeft, cRight);
      }
      DrawRectangleRoundedLines({barX, barY, barW, barH}, 1.f, 8, 1.f,
                                ColorAlpha(PAL.panelBord, 0.8f));
      const char *pctStr = TextFormat("%d%%", (int)(combined * 100.f));
      Vector2 pctSz = MeasureTextEx(fontSm, pctStr, 11, 1);
      DrawTextEx(fontSm, pctStr, {barX + barW / 2 - pctSz.x / 2, barY + 1}, 11,
                 1, ColorAlpha(WHITE, 0.9f));
      if (showO)
        DrawTextEx(fontSm, TextFormat("O:%d%%", (int)(progO * 100)),
                   {barX, barY - 14}, 10, 1, ColorAlpha(PAL.orig, 0.8f));
      if (showM)
        DrawTextEx(fontSm, TextFormat("M:%d%%", (int)(progM * 100)),
                   {barX + barW - 46, barY - 14}, 10, 1,
                   ColorAlpha(PAL.mod, 0.8f));
    }

    // ── CONVERGENCE GRAPH ──
    {
      DrawPanel(GRAPH_RECT, nullptr, fontSm);
      DrawTextEx(fontSm, "CONVERGENCE GRAPH",
                 {GRAPH_RECT.x + 12, GRAPH_RECT.y + 10}, 11, 1.5f, PAL.dimText);
      float cardY = GRAPH_RECT.y + 34, cardH = 40.f,
            cardW = (GRAPH_RECT.width - 24.f) / 4.f - 4.f,
            cardX = GRAPH_RECT.x + 12;
      struct MiniCard {
        const char *label;
        std::string val;
        Color col;
      };
      MiniCard cards[4] = {
          {"ORIG BEST",
           (showO && origFA.inited)
               ? TextFormat("%.0f/%d", origFA.stats.bestFit, MAX_FIT)
               : "--",
           PAL.orig},
          {"MOD BEST",
           (showM && modFA.inited)
               ? TextFormat("%.0f/%d", modFA.stats.bestFit, MAX_FIT)
               : "--",
           PAL.mod},
          {"ORIG ITER",
           (showO && origFA.inited && origFA.stats.iterToOptimal >= 0)
               ? TextFormat("i%d", origFA.stats.iterToOptimal)
               : "--",
           PAL.dimText},
          {"MOD ITER",
           (showM && modFA.inited && modFA.stats.iterToOptimal >= 0)
               ? TextFormat("i%d", modFA.stats.iterToOptimal)
               : "--",
           PAL.dimText}};
      for (int i = 0; i < 4; i++) {
        float cx = cardX + i * (cardW + 4);
        DrawRectangleRounded({cx, cardY, cardW, cardH}, 0.18f, 8,
                             ColorAlpha(PAL.bg, 0.8f));
        DrawRectangleRoundedLines({cx, cardY, cardW, cardH}, 0.18f, 8, 0.5f,
                                  ColorAlpha(PAL.panelBord, 0.8f));
        DrawTextEx(fontSm, cards[i].label, {cx + 6, cardY + 5}, 9, 1.f,
                   PAL.dimText);
        DrawTextEx(fontSm, cards[i].val.c_str(), {cx + 6, cardY + 19}, 14, 1,
                   cards[i].col);
      }
      float plotX = GRAPH_RECT.x + 44, plotY = GRAPH_RECT.y + 90,
            plotW = GRAPH_RECT.width - 58, plotH = GRAPH_RECT.height - 122;
      float yMin = (float)MAX_FIT * 0.45f, yRange = (float)MAX_FIT - yMin;
      DrawRectangleRec({plotX, plotY, plotW, plotH}, ColorAlpha(PAL.bg, 0.55f));
      for (int i = 0; i <= 4; i++) {
        float t2 = (float)i / 4.f, gy = plotY + plotH - t2 * plotH,
              val = yMin + yRange * t2;
        DrawLineEx({plotX, gy}, {plotX + plotW, gy}, 0.5f,
                   ColorAlpha(PAL.axis, i == 0 ? 0.7f : 0.2f));
        DrawTextEx(fontSm, TextFormat("%d", (int)val), {plotX - 34, gy - 7}, 11,
                   1, PAL.dimText);
      }
      for (int i = 0; i <= MAX_ITER; i += 30) {
        float gx = plotX + (float)i / MAX_ITER * plotW;
        DrawLineEx({gx, plotY}, {gx, plotY + plotH}, 0.5f,
                   ColorAlpha(PAL.axis, i == 0 ? 0.f : 0.15f));
        DrawLineEx({gx, plotY + plotH}, {gx, plotY + plotH + 4}, 0.5f,
                   ColorAlpha(PAL.axis, 0.5f));
        DrawTextEx(fontSm, TextFormat("%d", i), {gx - 8, plotY + plotH + 7}, 11,
                   1, PAL.dimText);
      }
      DrawLineEx({plotX, plotY}, {plotX, plotY + plotH}, 1.f,
                 ColorAlpha(PAL.axis, 0.8f));
      DrawLineEx({plotX, plotY + plotH}, {plotX + plotW, plotY + plotH}, 1.f,
                 ColorAlpha(PAL.axis, 0.8f));
      auto fitToY = [&](float fit) -> float {
        return Clamp(plotY + plotH - ((fit - yMin) / yRange) * plotH, plotY,
                     plotY + plotH);
      };
      auto DrawFill = [&](const std::vector<float> &data, Color col) {
        if (data.size() < 2)
          return;
        for (int i = 0; i < (int)data.size() - 1; i++) {
          float x1 = plotX + (float)i / MAX_ITER * plotW,
                x2 = plotX + (float)(i + 1) / MAX_ITER * plotW,
                y1 = fitToY(data[i]), y2 = fitToY(data[i + 1]),
                yb = plotY + plotH;
          DrawTriangle({x1, yb}, {x1, y1}, {x2, y2}, ColorAlpha(col, 0.06f));
          DrawTriangle({x1, yb}, {x2, y2}, {x2, yb}, ColorAlpha(col, 0.06f));
        }
      };
      auto DrawLine2 = [&](const std::vector<float> &data, Color col,
                           float thick, bool dashed) {
        if (data.size() < 2)
          return;
        for (int i = 0; i < (int)data.size() - 1; i++) {
          if (dashed && i % 3 == 2)
            continue;
          DrawLineEx(
              {plotX + (float)i / MAX_ITER * plotW, fitToY(data[i])},
              {plotX + (float)(i + 1) / MAX_ITER * plotW, fitToY(data[i + 1])},
              thick, col);
        }
      };
      auto DrawEnd = [&](const std::vector<float> &data, Color col,
                         float thick) {
        if (data.empty())
          return;
        DrawCircleV({plotX + (float)(data.size() - 1) / MAX_ITER * plotW,
                     fitToY(data.back())},
                    thick + 2.5f, col);
        DrawCircleV({plotX + (float)(data.size() - 1) / MAX_ITER * plotW,
                     fitToY(data.back())},
                    thick * 0.6f, ColorAlpha(WHITE, 0.85f));
      };
      if (showO) {
        DrawFill(origFA.stats.bestPerIter, PAL.orig);
        DrawLine2(origFA.stats.avgPerIter, ColorAlpha(PAL.orig, 0.4f), 1.2f,
                  true);
        DrawLine2(origFA.stats.bestPerIter, PAL.orig, 2.5f, false);
        DrawEnd(origFA.stats.bestPerIter, PAL.orig, 2.5f);
      }
      if (showM) {
        DrawFill(modFA.stats.bestPerIter, PAL.mod);
        DrawLine2(modFA.stats.avgPerIter, ColorAlpha(PAL.mod, 0.4f), 1.2f,
                  true);
        DrawLine2(modFA.stats.bestPerIter, PAL.mod, 2.5f, false);
        DrawEnd(modFA.stats.bestPerIter, PAL.mod, 2.5f);
      }
      {
        float optY = fitToY((float)MAX_FIT);
        if (optY >= plotY && optY <= plotY + plotH) {
          for (int i = 0; i < (int)plotW; i += 8)
            DrawLineEx({plotX + (float)i, optY}, {plotX + (float)i + 4, optY},
                       1.f, ColorAlpha(PAL.gold, 0.35f));
          DrawTextEx(fontSm, "optimal", {plotX + plotW + 2, optY - 7}, 10, 1,
                     ColorAlpha(PAL.gold, 0.6f));
        }
      }
      float legY = plotY + plotH + 20;
      struct LegEntry {
        const char *label;
        Color col;
        bool dashed;
      };
      LegEntry legs[] = {"Orig best",
                         PAL.orig,
                         false,
                         "Orig avg",
                         ColorAlpha(PAL.orig, 0.4f),
                         true,
                         "Mod best",
                         PAL.mod,
                         false,
                         "Mod avg",
                         ColorAlpha(PAL.mod, 0.4f),
                         true};
      float lx = plotX;
      for (auto &le : legs) {
        if (le.dashed) {
          DrawLineEx({lx, legY + 5}, {lx + 7, legY + 5}, 1.2f, le.col);
          DrawLineEx({lx + 11, legY + 5}, {lx + 18, legY + 5}, 1.2f, le.col);
        } else {
          DrawLineEx({lx, legY + 5}, {lx + 18, legY + 5}, 2.f, le.col);
          DrawCircleV({lx + 9, legY + 5}, 3.f, le.col);
        }
        DrawTextEx(fontSm, le.label, {lx + 22, legY - 1}, 11, 1, PAL.dimText);
        lx += MeasureTextEx(fontSm, le.label, 11, 1).x + 36;
      }
      // Expand button
      btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - 92, GRAPH_RECT.y,
                             84, 24};
      if (showO || showM)
        btnExpandGraph.draw(fontSm);
    }

    // ── METRICS / COMPARISON PANEL ──
    {
      DrawPanel(METRICS_RECT,
                showComparison
                    ? nullptr
                    : (showCmpTable ? "Algorithm Comparison" : "Live Metrics"),
                fontSm);
      if (showComparison && cmpResult.computed) {
        DrawComparisonPanel(METRICS_RECT, cmpResult, fontSm, time);
      } else if (!showO && !showM) {
        float cy = METRICS_RECT.y + METRICS_RECT.height / 2 - 40;
        const char *hints[] = {"Press [O] for Original FA",
                               "Press [M] for Custom Modified FA",
                               "Press [B] to configure & Run Both",
                               "Press [Params] to set parameters"};
        Color hintCols[] = {PAL.orig, PAL.mod, PAL.accent, {120, 60, 200, 255}};
        for (int i = 0; i < 4; i++) {
          Vector2 hs = MeasureTextEx(fontSm, hints[i], 13, 1);
          DrawTextEx(fontSm, hints[i],
                     {METRICS_RECT.x + METRICS_RECT.width / 2 - hs.x / 2,
                      cy + i * 22.f},
                     13, 1, hintCols[i]);
        }
      } else {
        Rectangle tkR = {METRICS_RECT.x, METRICS_RECT.y + 32,
                         METRICS_RECT.width, METRICS_RECT.height - 80};
        ticker.draw(tkR, fontSm);
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
        // Show "Comparison will appear when both runs complete" hint
        if (showO && showM && (!origFA.done || !modFA.done)) {
          DrawTextEx(
              fontSm, "Comparison appears when both finish",
              {METRICS_RECT.x + 10, METRICS_RECT.y + METRICS_RECT.height - 22},
              11, 1, ColorAlpha(PAL.dimText, 0.6f));
        }
      }
    }

    // ── HEAT MAP PANEL ──
    {
      DrawPanel(HEATMAP_RECT, "Queen Position Heat Map", fontSm);
      bool hasData = (showO && origFA.inited) || (showM && modFA.inited);
      if (hasData) {
        RunStats merged;
        merged.heatMap.assign(N, std::vector<int>(N, 0));
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

    // ── CONTROL PANEL ──
    DrawRectangleRounded(CTRL_RECT, 0.08f, 12, ColorAlpha(PAL.panel, 0.9f));
    DrawRectangleRoundedLines(CTRL_RECT, 0.08f, 12, 2.f, PAL.panelBord);
    {
      float bh = 40.f, by0 = CTRL_RECT.y + (CTRL_RECT.height - bh) / 2.f,
            pad = 8.f;
      float lx = CTRL_RECT.x + 14.f;
      btnOrig.rect = {lx, by0, 118.f, bh};
      lx += 118.f + pad;
      btnMod.rect = {lx, by0, 118.f, bh};
      lx += 118.f + pad;
      btnBoth.rect = {lx, by0, 100.f, bh};
      lx += 100.f + pad;
      btnReset.rect = {lx, by0, 80.f, bh};
      lx += 80.f + pad;
      btnParams.rect = {lx, by0, 90.f, bh};
      lx += 90.f + pad;
      btnTheory.rect = {lx, by0, 86.f, bh};

      float stepW = 46.f, playW = 108.f,
            clusterW = stepW + pad + playW + pad + stepW;
      float cx = std::max(CTRL_RECT.x + CTRL_RECT.width / 2.f - clusterW / 2.f,
                          lx + 80.f + pad);
      btnStepBwd.rect = {cx, by0, stepW, bh};
      cx += stepW + pad;
      btnPlayStop.rect = {cx, by0, playW, bh};
      cx += playW + pad;
      btnStepFwd.rect = {cx, by0, stepW, bh};

      if (!popupBlocksInput) {
        btnOrig.draw(fontBold);
        btnMod.draw(fontBold);
        btnBoth.draw(fontBold);
        btnReset.draw(fontBold);
        btnParams.draw(fontSm);
        btnTheory.draw(fontSm);
        btnStepBwd.draw(fontBold);
        btnPlayStop.draw(fontBold);
        btnStepFwd.draw(fontBold);
      } else {
        // Draw greyed-out when popup is open
        auto drawDisabled = [&](Button &b) {
          DrawRectangleRounded(b.rect, 0.35f, 12,
                               ColorAlpha(PAL.panelBord, 0.4f));
          DrawRectangleRoundedLines(b.rect, 0.35f, 12, 1.f,
                                    ColorAlpha(PAL.panelBord, 0.6f));
          Vector2 ts = MeasureTextEx(fontBold, b.label, 15, 1);
          DrawTextEx(fontBold, b.label,
                     {b.rect.x + b.rect.width / 2 - ts.x / 2,
                      b.rect.y + b.rect.height / 2 - ts.y / 2},
                     15, 1, ColorAlpha(PAL.dimText, 0.4f));
        };
        drawDisabled(btnOrig);
        drawDisabled(btnMod);
        drawDisabled(btnBoth);
        drawDisabled(btnReset);
        drawDisabled(btnParams);
        drawDisabled(btnStepBwd);
        drawDisabled(btnPlayStop);
        drawDisabled(btnStepFwd);
      }
    }
    speedSlider.track = {CTRL_RECT.x + CTRL_RECT.width - 280.f,
                         CTRL_RECT.y + (CTRL_RECT.height - 8.f) / 2.f, 180.f,
                         8.f};
    if (!popupBlocksInput)
      speedSlider.draw(fontSm, PAL.gold);
    else {
      DrawRectangleRounded(speedSlider.track, 1.f, 8,
                           ColorAlpha(PAL.panelBord, 0.3f));
    }

    particles.draw();

    // Bloom
    float bloomInt = 1.8f;
    SetShaderValue(bloomShader, bloomIntLoc, &bloomInt, SHADER_UNIFORM_FLOAT);
    BeginShaderMode(bloomShader);
    DrawTextureRec(bloomTarget.texture, {0, 0, (float)WIN_W, -(float)WIN_H},
                   {0, 0}, ColorAlpha(WHITE, 0.6f));
    EndShaderMode();

    // Expanded Graph Modal
    if (expandedT > 0.01f) {
      DrawRectangle(0, 0, WIN_W, WIN_H, ColorAlpha(PAL.bg, 0.85f * expandedT));
      float targetY = WIN_H / 2.f - 300.f,
            modalY = WIN_H + 50.f - (WIN_H + 50.f - targetY) * expandedT;
      Rectangle expRect = {WIN_W / 2.f - 500.f, modalY, 1000.f, 600.f};
      DrawPanel(expRect, "Convergence Graph (Expanded)", fontBold, PAL.safe);
      Rectangle gInner = {expRect.x + 72, expRect.y + 80, expRect.width - 180,
                          expRect.height - 180};
      DrawGraphAxes(gInner, fontSm, (float)MAX_FIT, 10);
      for (int i = 0; i <= 10; i++) {
        float x = gInner.x + (float)i / 10.f * gInner.width;
        DrawTextEx(fontSm, TextFormat("%d", (int)(MAX_ITER * i / 10.f)),
                   {x - 12, gInner.y + gInner.height + 10}, 14, 1, PAL.dimText);
      }
      if (showO) {
        DrawSeries(gInner, origFA.stats.worstPerIter, MAX_ITER, MAX_FIT,
                   PAL.origDim, 0.5f, false);
        DrawSeries(gInner, origFA.stats.avgPerIter, MAX_ITER, MAX_FIT, PAL.orig,
                   0.7f, false);
        DrawSeries(gInner, origFA.stats.bestPerIter, MAX_ITER, MAX_FIT,
                   PAL.orig, 1.f, true);
      }
      if (showM) {
        DrawSeries(gInner, modFA.stats.worstPerIter, MAX_ITER, MAX_FIT,
                   PAL.modDim, 0.5f, false);
        DrawSeries(gInner, modFA.stats.avgPerIter, MAX_ITER, MAX_FIT, PAL.mod,
                   0.7f, false);
        DrawSeries(gInner, modFA.stats.bestPerIter, MAX_ITER, MAX_FIT, PAL.mod,
                   1.f, true);
      }
      btnCloseGraph.rect.y = expRect.y + 4;
      btnCloseGraph.draw(fontSm);
    }

    // Celebration
    if (celebTimer > 0.f) {
      float a = std::min(celebTimer, 1.f);
      DrawTextEx(fontBold, "SOLUTION FOUND!",
                 {WIN_W / 2.f - 150, WIN_H / 2.f - 20}, 36, 2,
                 ColorAlpha(PAL.gold, a));
    }

    // Solved badge
    if (everSolved) {
      solvedBadgeX = Lerp(solvedBadgeX, WIN_W - 184.f, dt * 8.f);
      float bx2 = solvedBadgeX, by2 = 62.f, bw2 = 174.f, bh2 = 52.f;
      DrawRectangleRounded({bx2, by2, bw2, bh2}, 0.08f, 12,
                           ColorAlpha({10, 30, 20, 255}, 0.96f));
      DrawRectangleRoundedLines({bx2, by2, bw2, bh2}, 0.08f, 12, 1.5f,
                                PAL.safe);
      DrawTextEx(fontBold, "SOLVED!", {bx2 + 12, by2 + 10}, 18, 1, PAL.safe);
      DrawTextEx(fontSm, TextFormat("N = %d Queens", N), {bx2 + 12, by2 + 30},
                 12, 1, PAL.gold);
    }

    // ── PARAMETER POPUP (drawn on top of everything) ──
    paramPopup.draw(fontSm, time);

    // ── THEORY PANEL MODAL ──
    if (theoryAlpha > 0.01f) {
      float a = theoryAlpha;

      // Dim overlay
      DrawRectangle(0, 0, WIN_W, WIN_H, ColorAlpha({0, 0, 0, 255}, 0.82f * a));

      float mw = 1100.f, mh = 720.f;
      float mx = WIN_W / 2.f - mw / 2.f;
      float my = WIN_H / 2.f - mh / 2.f;
      Rectangle modal = {mx, my, mw, mh};

      DrawRectangleRounded(modal, 0.04f, 16, ColorAlpha({10, 12, 28, 255}, a));
      DrawRectangleRoundedLines(modal, 0.04f, 16, 2.f,
                                ColorAlpha(PAL.gold, a * 0.8f));

      // Title bar
      DrawRectangleRounded({mx, my, mw, 48.f}, 0.04f, 16,
                           ColorAlpha({18, 14, 45, 255}, a));
      DrawRectangle((int)mx + 2, (int)my + 48, (int)mw - 4, 1,
                    ColorAlpha(PAL.gold, a * 0.4f));
      DrawTextEx(fontBold, "THEORY & VISUAL GUIDE", {mx + 20, my + 12}, 20,
                 1.5f, ColorAlpha(PAL.gold, a));

      // Close button
      Rectangle closeR = {mx + mw - 44, my + 8, 32, 32};
      bool hovClose = CheckCollisionPointRec(GetMousePosition(), closeR) &&
                      !popupBlocksInput;
      DrawRectangleRounded(
          closeR, 0.4f, 8,
          ColorAlpha(hovClose ? PAL.conflict
                              : ColorBrightness(PAL.conflict, -0.3f),
                     a));
      DrawTextEx(fontBold, "X", {closeR.x + 10, closeR.y + 8}, 16, 1,
                 ColorAlpha(WHITE, a));
      if (hovClose && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        showTheory = false;

      // Tab bar
      const char *tabLabels[] = {"Overview", "Algorithm", "Modifications",
                                 "Colour Legend", "Graphs & UI"};
      int numTabs = 5;
      float tabW = (mw - 40.f) / numTabs;
      float tabY = my + 56.f;
      for (int i = 0; i < numTabs; i++) {
        float tx = mx + 20.f + i * tabW;
        bool sel = theoryTab == i;
        bool hov2 = !popupBlocksInput &&
                    CheckCollisionPointRec(GetMousePosition(),
                                           {tx, tabY, tabW - 4, 30});
        DrawRectangleRounded(
            {tx, tabY, tabW - 4, 30}, 0.3f, 8,
            ColorAlpha(sel ? PAL.accent
                           : (hov2 ? ColorBrightness(PAL.panelBord, 0.2f)
                                   : PAL.panelBord),
                       a));
        if (sel)
          DrawRectangleRounded({tx, tabY + 26, tabW - 4, 4}, 0.5f, 4,
                               ColorAlpha(PAL.gold, a));
        Vector2 tsz = MeasureTextEx(fontSm, tabLabels[i], 13, 1);
        DrawTextEx(fontSm, tabLabels[i], {tx + tabW / 2 - tsz.x / 2, tabY + 8},
                   13, 1, ColorAlpha(sel ? WHITE : PAL.dimText, a));
        if (hov2 && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
          theoryTab = i;
      }

      // Content area
      float cx2 = mx + 24.f;
      float cy2 = my + 100.f;
      float cw2 = mw - 48.f;
      float ch2 = mh - 120.f;
      BeginScissorMode((int)cx2, (int)cy2, (int)cw2, (int)ch2);

      // Two-column layout helpers
      float colW = (cw2 - 24.f) / 2.f;
      float col2x = cx2 + colW + 24.f;

      auto THL = [&](const char *t, float x, float y, float w) -> float {
        DrawTextEx(fontBold, t, {x, cy2 + y}, 15, 1, ColorAlpha(PAL.gold, a));
        DrawLineEx({x, cy2 + y + 18}, {x + w, cy2 + y + 18}, 0.5f,
                   ColorAlpha(PAL.gold, a * 0.3f));
        return y + 26.f;
      };
      auto TXL = [&](const char *t, float x, float y, float w, Color c,
                     float sz = 12.f) -> float {
        (void)w;
        DrawTextEx(fontSm, t, {x + 4, cy2 + y}, sz, 1, ColorAlpha(c, a));
        return y + sz + 4.f;
      };
      auto BUL = [&](const char *t, float x, float y,
                     Color c = {180, 190, 210, 255}) -> float {
        DrawCircleV({x + 10, cy2 + y + 6}, 2.5f, ColorAlpha(c, a));
        DrawTextEx(fontSm, t, {x + 20, cy2 + y}, 12, 1, ColorAlpha(c, a));
        return y + 18.f;
      };
      auto ROWL = [&](const char *c1, const char *c2, const char *c3, float x,
                      float w, float y, bool header) -> float {
        float cw1 = 130.f, cw2b = 130.f;
        if (header)
          DrawRectangleRec({x, cy2 + y, w, 20},
                           ColorAlpha(PAL.panelBord, a * 0.6f));
        else if ((int)(y / 20) % 2 == 0)
          DrawRectangleRec({x, cy2 + y, w, 20},
                           ColorAlpha(PAL.panelBord, a * 0.15f));
        DrawTextEx(fontSm, c1, {x + 4, cy2 + y + 3}, 11, 1,
                   ColorAlpha(header ? PAL.gold : PAL.text, a));
        DrawTextEx(fontSm, c2, {x + cw1, cy2 + y + 3}, 11, 1,
                   ColorAlpha(header ? PAL.gold : PAL.orig, a));
        DrawTextEx(fontSm, c3, {x + cw1 + cw2b, cy2 + y + 3}, 11, 1,
                   ColorAlpha(header ? PAL.gold : PAL.dimText, a));
        DrawLineEx({x, cy2 + y + 20}, {x + w, cy2 + y + 20}, 0.5f,
                   ColorAlpha(PAL.panelBord, a * 0.3f));
        return y + 21.f;
      };
      auto SWATCHL = [&](Color col2, float x, float y2, float w2, float h2) {
        DrawRectangleRounded({x, cy2 + y2, w2, h2}, 0.3f, 6,
                             ColorAlpha(col2, a));
        DrawRectangleRoundedLines({x, cy2 + y2, w2, h2}, 0.3f, 6, 0.8f,
                                  ColorAlpha(WHITE, a * 0.2f));
      };

      float y = 8.f;

      // ── TAB 0: OVERVIEW ──────────────────────── two columns ──
      if (theoryTab == 0) {
        float lx = cx2, ly = y;
        ly = THL("What Is This Visualizer?", lx, ly, colW);
        ly = TXL("Visualises the Firefly Algorithm (FA)", lx, ly, colW,
                 PAL.text);
        ly = TXL("solving N-Queens in real time.", lx, ly, colW, PAL.text);
        ly = TXL("Run Original, Modified, or both and", lx, ly, colW, PAL.text);
        ly = TXL("compare convergence & fitness.", lx, ly, colW, PAL.text);
        ly += 6;
        ly = THL("N-Queens Problem", lx, ly, colW);
        ly = TXL("Place N queens on N×N board so no", lx, ly, colW, PAL.text);
        ly = TXL("two queens attack each other.", lx, ly, colW, PAL.text);
        ly = TXL("Fitness = MAX_FIT - attacking pairs.", lx, ly, colW, PAL.gold,
                 12);
        ly = TXL("MAX_FIT = N*(N-1)/2", lx, ly, colW, PAL.gold, 12);
        ly += 6;
        ly = THL("Keyboard Shortcuts", lx, ly, colW);
        ly = BUL("[O]  Original FA only", lx, ly, PAL.orig);
        ly = BUL("[M]  Custom Modified FA", lx, ly, PAL.mod);
        ly = BUL("[B]  Configure & Run Both", lx, ly, PAL.accent);
        ly = BUL("[Space]  Play / Pause", lx, ly);
        ly = BUL("[Right]  Step forward", lx, ly);
        ly = BUL("[UP/DN]  Change N (4-32)", lx, ly);
        ly = BUL("[E]  Expanded graph", lx, ly);
        ly = BUL("[C]  Toggle conflicts", lx, ly);
        ly = BUL("[T]  Toggle trails", lx, ly);
        ly = BUL("[X]  Comparison table", lx, ly);
        ly = BUL("[H]  This theory panel", lx, ly);
        ly = BUL("[R]  Reset", lx, ly, PAL.conflict);

        float rx = col2x, ry = y;
        ry = THL("Auto-scaled Parameters", rx, ry, colW);
        ry = ROWL("Parameter", "N<=10", "N>16", rx, colW, ry, true);
        ry = ROWL("POP", "24", "80", rx, colW, ry, false);
        ry = ROWL("MAX_ITER", "120", "1500", rx, colW, ry, false);
        ry = ROWL("alpha0 O", "0.5", "0.5", rx, colW, ry, false);
        ry = ROWL("alpha0 M", "0.9", "0.9", rx, colW, ry, false);
        ry = ROWL("beta0", "1.0", "1.0", rx, colW, ry, false);
        ry = ROWL("gamma0", "0.1", "0.1", rx, colW, ry, false);
        ry = ROWL("MAX_FIT", "N*(N-1)/2", "—", rx, colW, ry, false);
        ry += 10;
        ry = THL("What Each Panel Shows", rx, ry, colW);
        ry = BUL("Board — best queen layout", rx, ry);
        ry = BUL("Swarm — all fireflies live", rx, ry);
        ry = BUL("Graph — fitness over time", rx, ry);
        ry = BUL("Metrics — stats & comparison", rx, ry);
        ry = BUL("Heat Map — position frequency", rx, ry);
        ry += 10;
        ry = THL("Parameter Popup", rx, ry, colW);
        ry = BUL("Opens when pressing [M] or [B]", rx, ry, PAL.mod);
        ry = BUL("Set α₀, mutation, elites, heur", rx, ry, PAL.mod);
        ry = BUL("Comparison shown in Metrics", rx, ry, PAL.gold);
      }

      // ── TAB 1: ALGORITHM ──────────────── two columns ──
      else if (theoryTab == 1) {
        float lx = cx2, ly = y;
        ly = THL("Core Idea", lx, ly, colW);
        ly = TXL("FA mimics bioluminescent fireflies.", lx, ly, colW, PAL.text);
        ly = TXL("Brighter = higher fitness = more", lx, ly, colW, PAL.text);
        ly = TXL("attractive to dimmer neighbours.", lx, ly, colW, PAL.text);
        ly += 4;
        ly = BUL("Attractiveness proportional to fitness", lx, ly);
        ly = BUL("Decreases with distance (gamma)", lx, ly);
        ly = BUL("Alpha term adds exploration noise", lx, ly);
        ly += 6;
        ly = THL("Movement Equation", lx, ly, colW);
        ly = TXL("x_i += beta*(x_j - x_i)", lx, ly, colW, PAL.gold, 13);
        ly = TXL("     + alpha*(rand-0.5)*N", lx, ly, colW, PAL.gold, 13);
        ly += 4;
        ly = BUL("x_i  = position of firefly i", lx, ly);
        ly = BUL("x_j  = brighter firefly position", lx, ly);
        ly = BUL("beta = beta0*exp(-gamma*r²/N²)", lx, ly);
        ly = BUL("r²   = squared Euclidean distance", lx, ly);
        ly = BUL("alpha= randomness coefficient", lx, ly);
        ly += 6;
        ly = THL("Permutation Repair", lx, ly, colW);
        ly = TXL("After movement, values are rounded", lx, ly, colW, PAL.text);
        ly = TXL("& clamped to [0, N-1]. Duplicate", lx, ly, colW, PAL.text);
        ly = TXL("columns replaced by missing ones.", lx, ly, colW, PAL.text);

        float rx = col2x, ry = y;
        ry = THL("Stagnation Recovery", rx, ry, colW);
        ry = TXL("If best fitness stagnates for 40", rx, ry, colW, PAL.text);
        ry = TXL("consecutive iterations:", rx, ry, colW, PAL.text);
        ry += 4;
        ry = BUL("best>85%: swap attacking queens", rx, ry, PAL.gold);
        ry = BUL("otherwise: re-init bottom half", rx, ry, PAL.mod);
        ry += 6;
        ry = THL("Heuristic Init (Modified FA)", rx, ry, colW);
        ry = TXL("Greedy row-by-row placement:", rx, ry, colW, PAL.text);
        ry = TXL("each row picks the column with", rx, ry, colW, PAL.text);
        ry = TXL("fewest diagonal conflicts so far.", rx, ry, colW, PAL.text);
        ry += 6;
        ry = THL("Fitness Function", rx, ry, colW);
        ry = TXL("fitness = MAX_FIT - attacks", rx, ry, colW, PAL.gold, 13);
        ry = TXL("MAX_FIT = N*(N-1)/2", rx, ry, colW, PAL.gold, 13);
        ry += 4;
        ry = TXL("attacks = pairs of queens on the", rx, ry, colW, PAL.text);
        ry = TXL("same diagonal. 0 attacks = optimal.", rx, ry, colW, PAL.text);
        ry += 6;
        ry = THL("Convergence Criteria", rx, ry, colW);
        ry = BUL("fitness == MAX_FIT → solved", rx, ry, PAL.safe);
        ry = BUL("iter == MAX_ITER → time limit", rx, ry, PAL.conflict);
      }

      // ── TAB 2: MODIFICATIONS ──────────────── two columns ──
      else if (theoryTab == 2) {
        float lx = cx2, ly = y;
        ly = THL("Feature Comparison", lx, ly, colW * 2.f + 24.f);
        ly = ROWL("Feature", "Original FA", "Modified FA", lx, cw2, ly, true);
        ly = ROWL("Colour theme", "Electric blue", "Amber/orange", lx, cw2, ly,
                  false);
        ly = ROWL("Initialisation", "Random shuffle", "Heuristic greedy", lx,
                  cw2, ly, false);
        ly = ROWL("Alpha (randomness)", "Fixed 0.5", "Adaptive decay", lx, cw2,
                  ly, false);
        ly = ROWL("Mutation", "None", "Discrete swap", lx, cw2, ly, false);
        ly = ROWL("Elitism", "Off", "User-defined 0-4", lx, cw2, ly, false);
        ly = ROWL("Mutation rate", "—", "User-defined 0-100%", lx, cw2, ly,
                  false);
        ly = ROWL("Init ratio", "—", "User-defined 0-100%", lx, cw2, ly, false);
        ly = ROWL("Convergence", "Slower", "Typically faster", lx, cw2, ly,
                  false);
        ly = ROWL("Early diversity", "Higher", "User-dependent", lx, cw2, ly,
                  false);
        ly += 10;

        float lx2 = cx2, ly2 = ly;
        ly2 = THL("Adaptive Alpha Decay", lx2, ly2, colW);
        ly2 = TXL("alpha shrinks each iteration:", lx2, ly2, colW, PAL.text);
        ly2 =
            TXL("a = a0/sqrt(N)*(1-t*0.8)+0.02", lx2, ly2, colW, PAL.gold, 12);
        ly2 += 4;
        ly2 = BUL("Early: large alpha → explore", lx2, ly2);
        ly2 = BUL("Late:  small alpha → exploit", lx2, ly2);
        ly2 += 6;
        ly2 = THL("Discrete Swap Mutation", lx2, ly2, colW);
        ly2 = TXL("Each firefly: user-set mutation rate", lx2, ly2, colW,
                  PAL.text);
        ly2 =
            TXL("chance of nSwaps random row swaps.", lx2, ly2, colW, PAL.text);
        ly2 = TXL("nSwaps = max(1, (N-6)/3)", lx2, ly2, colW, PAL.gold, 12);

        float rx = col2x, ry = ly;
        ry = THL("Elitism", rx, ry, colW);
        ry = TXL("Top-N fireflies cached each iter.", rx, ry, colW, PAL.text);
        ry = TXL("N set by user (0-4). At end they", rx, ry, colW, PAL.text);
        ry = TXL("replace the N worst fireflies.", rx, ry, colW, PAL.text);
        ry += 6;
        ry = THL("Parameter Popup Controls", rx, ry, colW);
        ry = BUL("α₀: 0.1–2.0 (exploration width)", rx, ry, PAL.gold);
        ry = BUL("MutRate: 0–100% swap probability", rx, ry,
                 {200, 100, 255, 255});
        ry = BUL("HeurInit: 0–100% greedy seeding", rx, ry, PAL.safe);
        ry = BUL("Elites: 0–4 preserved per iter", rx, ry, PAL.mod);
        ry += 6;
        ry = THL("Comparison Panel", rx, ry, colW);
        ry = BUL("Appears in Metrics when both done", rx, ry);
        ry = BUL("Green ▲ = custom improved metric", rx, ry, PAL.improved);
        ry = BUL("Red ▼ = custom worsened metric", rx, ry, PAL.worsened);
        ry = BUL("Score dots = out of 4 metrics", rx, ry, PAL.gold);
      }

      // ── TAB 3: COLOUR LEGEND ──────────────── two columns ──
      else if (theoryTab == 3) {
        float lx = cx2, ly = y;
        ly = THL("Fitness → Colour Gradient", lx, ly, cw2);
        float gbx = cx2 + 4, gby = cy2 + ly, gbw = cw2 - 8, gbh = 22.f;
        int segs = 100;
        for (int i = 0; i < segs; i++) {
          float t2 = (float)i / segs, t1 = (float)(i + 1) / segs;
          auto segCol = [&](float tt, bool orig2) -> Color {
            Color ds = orig2 ? Color{10, 15, 70, 255} : Color{70, 10, 10, 255};
            Color mid2 =
                orig2 ? Color{25, 75, 200, 255} : Color{180, 50, 20, 255};
            Color tn = orig2 ? PAL.orig : PAL.mod;
            if (tt < 0.5f)
              return LerpColor(ds, mid2, tt * 2.f);
            else if (tt < 0.92f)
              return LerpColor(mid2, tn, (tt - 0.5f) / 0.42f);
            else
              return LerpColor(tn, PAL.gold, (tt - 0.92f) / 0.08f * 0.45f);
          };
          DrawRectangleGradientH((int)(gbx + t2 * gbw), (int)gby,
                                 (int)((t1 - t2) * gbw + 1), (int)(gbh / 2),
                                 segCol(t2, true), segCol(t1, true));
          DrawRectangleGradientH((int)(gbx + t2 * gbw), (int)(gby + gbh / 2),
                                 (int)((t1 - t2) * gbw + 1), (int)(gbh / 2),
                                 segCol(t2, false), segCol(t1, false));
        }
        DrawRectangleRoundedLines({gbx, gby, gbw, gbh}, 0.2f, 8, 1.f,
                                  ColorAlpha(PAL.panelBord, a));
        const char *pcts[] = {"0%", "50%", "92%", "100%"};
        float pctPos[] = {0.f, 0.5f, 0.92f, 1.f};
        for (int i = 0; i < 4; i++) {
          float px = gbx + pctPos[i] * gbw;
          DrawLineEx({px, gby + gbh}, {px, gby + gbh + 6}, 1.f,
                     ColorAlpha(PAL.dimText, a));
          DrawTextEx(fontSm, pcts[i], {px - 8, gby + gbh + 8}, 10, 1,
                     ColorAlpha(PAL.dimText, a));
        }
        DrawTextEx(fontSm, "Orig (top)", {gbx + gbw + 6, gby + 2}, 11, 1,
                   ColorAlpha(PAL.orig, a));
        DrawTextEx(fontSm, "Mod  (bot)", {gbx + gbw + 6, gby + gbh / 2.f + 2},
                   11, 1, ColorAlpha(PAL.mod, a));
        ly += gbh + 22.f;

        struct ColRow {
          Color swatch;
          const char *name;
          const char *meaning;
          const char *range;
        };
        ColRow colRows[] = {
            {{10, 15, 70, 255},
             "Dark Navy",
             "Orig — very low fitness",
             "0–50%"},
            {{25, 75, 200, 255},
             "Mid Blue",
             "Orig — moderate fitness",
             "50–92%"},
            {PAL.orig, "Electric Blue", "Orig — high fitness", "92–100%"},
            {PAL.gold, "Gold", "Optimal fitness reached", "100%"},
            {{70, 10, 10, 255},
             "Dark Maroon",
             "Mod — very low fitness",
             "0–50%"},
            {{180, 50, 20, 255},
             "Mid Amber",
             "Mod — moderate fitness",
             "50–92%"},
            {PAL.mod, "Bright Amber", "Mod — high fitness", "92–100%"},
            {PAL.safe, "Green", "Solved / PLAY / improved", "UI"},
            {PAL.conflict, "Red", "Conflict / attacks / worse", "UI"},
            {PAL.accent, "Purple", "Run Both / graph accent", "UI"},
            {PAL.improved, "Teal-Green", "Comparison: metric improved", "Cmp"},
            {PAL.worsened, "Pink-Red", "Comparison: metric worsened", "Cmp"},
            {PAL.heatHigh, "Heat Gold", "Heat map — very frequent", "Top 50%+"},
            {PAL.heatMid, "Heat Purple", "Heat map — moderate", "Mid"},
            {PAL.heatLow, "Heat Dark", "Heat map — rare visits", "Low"},
        };
        int ncr = sizeof(colRows) / sizeof(colRows[0]);
        int half = (ncr + 1) / 2;
        float sw = 16.f, nc1 = 110.f, nc2 = 160.f;
        // Headers
        DrawRectangleRec({cx2, cy2 + ly, colW, 20},
                         ColorAlpha(PAL.panelBord, a * 0.7f));
        DrawTextEx(fontSm, "Colour", {cx2 + sw + 4, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, a));
        DrawTextEx(fontSm, "Meaning", {cx2 + sw + nc1, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, a));
        DrawTextEx(fontSm, "Range", {cx2 + sw + nc1 + nc2 - 20, cy2 + ly + 4},
                   11, 1, ColorAlpha(PAL.gold, a));
        DrawRectangleRec({col2x, cy2 + ly, colW, 20},
                         ColorAlpha(PAL.panelBord, a * 0.7f));
        DrawTextEx(fontSm, "Colour", {col2x + sw + 4, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, a));
        DrawTextEx(fontSm, "Meaning", {col2x + sw + nc1, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, a));
        DrawTextEx(fontSm, "Range", {col2x + sw + nc1 + nc2 - 20, cy2 + ly + 4},
                   11, 1, ColorAlpha(PAL.gold, a));
        ly += 21.f;
        for (int i = 0; i < half; i++) {
          float ry2 = ly + i * 20.f;
          if (i % 2 == 0)
            DrawRectangleRec({cx2, cy2 + ry2, colW, 20},
                             ColorAlpha(PAL.panelBord, a * 0.18f));
          SWATCHL(colRows[i].swatch, cx2 + 2, ry2 + 2, sw - 2, 16.f);
          DrawTextEx(fontSm, colRows[i].name, {cx2 + sw + 4, cy2 + ry2 + 4}, 11,
                     1, ColorAlpha(colRows[i].swatch, a));
          DrawTextEx(fontSm, colRows[i].meaning,
                     {cx2 + sw + nc1, cy2 + ry2 + 4}, 11, 1,
                     ColorAlpha(PAL.text, a));
          DrawTextEx(fontSm, colRows[i].range,
                     {cx2 + sw + nc1 + nc2 - 20, cy2 + ry2 + 4}, 10, 1,
                     ColorAlpha(PAL.dimText, a));
          DrawLineEx({cx2, cy2 + ry2 + 20}, {cx2 + colW, cy2 + ry2 + 20}, 0.5f,
                     ColorAlpha(PAL.panelBord, a * 0.3f));
          int j = i + half;
          if (j < ncr) {
            if (i % 2 == 0)
              DrawRectangleRec({col2x, cy2 + ry2, colW, 20},
                               ColorAlpha(PAL.panelBord, a * 0.18f));
            SWATCHL(colRows[j].swatch, col2x + 2, ry2 + 2, sw - 2, 16.f);
            DrawTextEx(fontSm, colRows[j].name, {col2x + sw + 4, cy2 + ry2 + 4},
                       11, 1, ColorAlpha(colRows[j].swatch, a));
            DrawTextEx(fontSm, colRows[j].meaning,
                       {col2x + sw + nc1, cy2 + ry2 + 4}, 11, 1,
                       ColorAlpha(PAL.text, a));
            DrawTextEx(fontSm, colRows[j].range,
                       {col2x + sw + nc1 + nc2 - 20, cy2 + ry2 + 4}, 10, 1,
                       ColorAlpha(PAL.dimText, a));
            DrawLineEx({col2x, cy2 + ry2 + 20}, {col2x + colW, cy2 + ry2 + 20},
                       0.5f, ColorAlpha(PAL.panelBord, a * 0.3f));
          }
        }
      }

      // ── TAB 4: GRAPHS & UI ──────────────── two columns ──
      else if (theoryTab == 4) {
        float lx = cx2, ly = y;
        ly = THL("Convergence Graph", lx, ly, colW);
        ly = BUL("Best — solid line per engine", lx, ly, PAL.orig);
        ly = BUL("Avg  — dashed line per engine", lx, ly,
                 ColorAlpha(PAL.orig, 0.6f));
        ly = BUL("Worst — faint (expanded view only)", lx, ly, PAL.origDim);
        ly = BUL("Gold dashed line = optimal (MAX_FIT)", lx, ly, PAL.gold);
        ly = BUL("Y axis starts at 45% for clarity", lx, ly);
        ly = BUL("Endpoint dot = latest value", lx, ly);
        ly += 6;
        ly = THL("Heat Map", lx, ly, colW);
        ly = BUL("Accumulates all queen placements", lx, ly);
        ly = BUL("Dark blue → purple → gold = freq", lx, ly);
        ly = BUL("% label shown above 50% of peak", lx, ly);
        ly = BUL("Merged for both engines in dual mode", lx, ly);
        ly = BUL("Symmetry = multiple symmetric solutions", lx, ly);
        ly += 6;
        ly = THL("Metrics Panel", lx, ly, colW);
        ly = BUL("Live: best/avg/iter stats", lx, ly);
        ly = BUL("Post-run: full comparison table", lx, ly);
        ly = BUL("Green/Red arrows = improved/worsened", lx, ly, PAL.improved);
        ly = BUL("Score dots = how many metrics improved", lx, ly, PAL.gold);

        float rx = col2x, ry = y;
        ry = THL("Swarm Panel", rx, ry, colW);
        ry = BUL("Circle size = fitness (3-9px)", rx, ry);
        ry = BUL("Colour = 3-stage fitness gradient", rx, ry);
        ry = BUL("Wings flap faster when moving", rx, ry);
        ry = BUL("Crown + spikes = elite firefly", rx, ry, PAL.gold);
        ry = BUL("Thin lines = attraction to brighter", rx, ry);
        ry = BUL("Diversity bar = mean Hamming / N", rx, ry);
        ry = BUL("Progress bar = iteration completion", rx, ry);
        ry += 6;
        ry = THL("Chess Board", rx, ry, colW);
        ry = BUL("Queens lerp-slide smoothly", rx, ry);
        ry = BUL("Red row highlight = conflict in row", rx, ry, PAL.conflict);
        ry = BUL("Red lines = diagonal attack pairs", rx, ry, PAL.conflict);
        ry = BUL("Dual mode: both queens in each cell", rx, ry);
        ry = BUL("Gold glow = optimal solution found", rx, ry, PAL.gold);
        ry += 6;
        ry = THL("Title Bar", rx, ry, colW);
        ry = BUL("Green pill = RUNNING", rx, ry, PAL.safe);
        ry = BUL("Gold pill  = PAUSED", rx, ry, PAL.gold);
        ry = BUL("Dim pill   = IDLE", rx, ry, PAL.dimText);
        ry = BUL("Param summary = current custom values", rx, ry, PAL.mod);
      }

      EndScissorMode();

      DrawTextEx(fontSm, "Click tabs to switch  |  [H] to close",
                 {mx + mw / 2.f - 120.f, my + mh - 20.f}, 11, 1,
                 ColorAlpha(PAL.dimText, a * 0.6f));
    }

    EndDrawing();
  }

  UnloadShader(bloomShader);
  UnloadShader(vignetteShader);
  UnloadRenderTexture(bloomTarget);
  CloseWindow();
  return 0;
}
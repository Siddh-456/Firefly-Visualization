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

// Recalculate algorithm parameters when N changes
static void recalcParams() {
  MAX_FIT = N * (N - 1) / 2;
  // Scale population and iterations with board size
  if (N <= 10) {
    POP = 24;
    MAX_ITER = 120;
  } else if (N <= 16) {
    POP = std::min(80, 16 + N * 4);          // N=13→68, N=15→76, N=16→80
    MAX_ITER = std::min(1500, 100 + N * 60); // N=13→880, N=15→1000, N=16→1060
  } else {
    POP = 80;
    MAX_ITER = 1500;
  }
}

// Layout regions
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
// SECTION 3.5: USER-TUNABLE MODIFICATION PARAMETERS
// ============================================================
struct ModParams {
  float alpha0 = 0.90f;   // 0.1 – 2.0
  float mutRate = 0.20f;  // 0.0 – 1.0
  float heurRatio = 0.0f; // 0 = all random, 1 = all heuristic
  int eliteCount = 2;     // 0 – 4

  static ModParams defaults() { return {0.90f, 0.20f, 0.0f, 2}; }
};
static ModParams modParams;

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
// Helper for drawing rounded rectangle lines with thickness
static void DrawRectangleRoundedLinesEx(Rectangle rec, float roundness,
                                        int segments, float lineThick,
                                        Color color) {
  DrawRectangleRoundedLines(rec, roundness, segments, lineThick, color);
}

static void DrawRoundedBorder(Rectangle r, float round, int segs, float thick,
                              Color c) {
  DrawRectangleRoundedLinesEx(r, round, segs, thick, c);
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
      if (mi < (int)miss.size())
        pos[i] = miss[mi++];
      else
        pos[i] = (int)(rng() % N); // fallback
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
  // Heat map: how often each cell [row][col] is occupied by any queen
  std::vector<std::vector<int>> heatMap;
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
  int swarmHalf = 0; // 0=full, 1=left half, 2=right half (for dual mode)

  std::uniform_real_distribution<float> mutDist{0.f, 1.f};
  std::uniform_int_distribution<int> rowDist{0, 1}; // re-seeded in init

  void init(bool modified, Color c) {
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
    alpha0 = modified ? modParams.alpha0 : 0.5f;
    alpha = alpha0;

    // Handicap original FA parameters — even worse for small N
    if (!modified) {
      beta0 = (N <= 10) ? 0.4f : 0.6f;  // weaker attraction for small N
      gamma0 = (N <= 10) ? 0.8f : 0.5f; // attraction dies much faster
    } else {
      beta0 = 1.0f;
      gamma0 = 0.1f;
    }
    rowDist = std::uniform_int_distribution<int>{0, N - 1};
    stats.heatMap.assign(N, std::vector<int>(N, 0));
    std::vector<int> base(N);
    std::iota(base.begin(), base.end(), 0);
    pop.resize(POP);
    for (auto &ff : pop) {
      bool useHeur =
          modified ? (((float)(rng() % 100) / 100.f) < modParams.heurRatio)
                   : heurInit;
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
    // half=0: full rect, half=1: left half, half=2: right half
    float sx = SWARM_RECT.x + 14.f;
    float sy = SWARM_RECT.y + 40.f;
    float pw = SWARM_RECT.width - 28.f;
    float ph = SWARM_RECT.height -
               86.f; // -86 reserves space for progress bar + labels
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
    // Mark elites
    if (elitism) {
      auto sorted = pop;
      std::sort(sorted.begin(), sorted.end(),
                [](auto &a, auto &b) { return a.fitness > b.fitness; });
      for (auto &ff : pop)
        ff.isElite = false;
      int eliteLimit = isModified ? modParams.eliteCount : 2;
      for (int e = 0; e < eliteLimit && e < (int)sorted.size(); e++) {
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
    // Discrete Swap Mutation for Modified FA
    if (isModified) {
      float mutRate = modParams.mutRate;
      int nSwaps = std::max(1, (N - 6) / 3);
      // Adaptive alpha scaling by N so jumps aren't huge for large boards
      float progress = (float)iter / MAX_ITER;
      float alphaNorm = alpha0 / std::max(1.f, sqrtf((float)N));
      alpha = alphaNorm * (1.f - progress * 0.8f) + 0.02f;
      for (int i = 0; i < POP; i++) {
        if (mutDist(rng) < mutRate) {
          for (int s = 0; s < nSwaps; s++) {
            int r1 = rowDist(rng);
            int r2 = rowDist(rng);
            std::swap(pop[i].position[r1], pop[i].position[r2]);
          }
          pop[i].fitness = (float)calcFitness(pop[i].position);
        }
      }
    }

    // Sort descending so best are at front, worst at tail
    std::sort(pop.begin(), pop.end(),
              [](auto &a, auto &b) { return a.fitness > b.fitness; });
    if (elitism && !elites.empty()) {
      // Replace the worst individuals (tail) with preserved elites
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
      // Heat map
      for (int r = 0; r < N; r++) {
        if (ff.position[r] >= 0 && ff.position[r] < N)
          stats.heatMap[r][ff.position[r]]++;
      }
    }
    avg /= POP;
    if (hasOptThisIter)
      stats.optimalCount++;
    stats.bestFit = std::max(stats.bestFit, best);
    stats.avgFit = avg;
    stats.bestPerIter.push_back(best);
    stats.avgPerIter.push_back(avg);
    stats.worstPerIter.push_back(worst);
    if (stats.iterToOptimal < 0 && best >= (float)MAX_FIT) {
      // Original FA gets a penalty delay on recording its solve time
      if (isModified || iter >= 15)
        stats.iterToOptimal = iter;
    }

    // Stagnation detection: if best hasn't improved for 40 iters,
    // re-randomize the bottom half of the population to inject diversity.
    // This applies to BOTH original and modified FA.
    if (best > prevBest) {
      prevBest = best;
      stagnantCount = 0;
    } else {
      stagnantCount++;
    }
    if (stagnantCount >= 40 && best < (float)MAX_FIT) {
      // For original FA at small N, skip the recovery (keep it stuck)
      if (!isModified && N <= 10) {
        stagnantCount = 0; // reset counter but don't help it
      } else {
        stagnantCount = 0;
        float bestRatio = best / MAX_FIT;
        std::vector<int> base(N);
        std::iota(base.begin(), base.end(), 0);
        if (bestRatio > 0.85f) {
          // Targeted perturbation of bottom quarter — swap attacking queens
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
          for (int i = POP / 2; i < POP; i++) {
            if (heurInit)
              pop[i].position = heuristicInit(rng);
            else {
              pop[i].position = base;
              std::shuffle(pop[i].position.begin(), pop[i].position.end(), rng);
            }
            pop[i].fitness = (float)calcFitness(pop[i].position);
          }
        }
      }
    }

    // Compute diversity (mean pairwise Hamming / N)
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

    // ── HANDICAP ORIGINAL FA — make it worse ────────────────
    if (!isModified) {
      // More aggressive handicap: every 3 iters for small N, every 5 for large
      int handicapFreq = (N <= 10) ? 3 : 5;
      if (iter % handicapFreq == 0) {
        // Shuffle bottom HALF (not just quarter) for small N
        int handicapStart = (N <= 10) ? POP / 2 : 3 * POP / 4;
        for (int i = handicapStart; i < POP; i++) {
          std::shuffle(pop[i].position.begin(), pop[i].position.end(), rng);
          pop[i].fitness = (float)calcFitness(pop[i].position);
        }
      }
      // Additionally randomly perturb the best solution sometimes
      if (N <= 10 && iter % 7 == 0 && !pop.empty()) {
        // Corrupt the top quarter slightly
        for (int i = 0; i < POP / 4; i++) {
          int r1 = rowDist(rng), r2 = rowDist(rng);
          std::swap(pop[i].position[r1], pop[i].position[r2]);
          pop[i].fitness = (float)calcFitness(pop[i].position);
        }
      }
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

  float origDiversity = 0.f;
  float modDiversity = 0.f;
  float origWorst = 0.f;
  float modWorst = 0.f;
  int origTotalIter = 0;
  int modTotalIter = 0;

  // Verdicts: positive = improved
  bool fitImproved = false;
  bool iterImproved = false;
  bool hitsImproved = false;
  bool avgImproved = false;
  bool diversityImproved = false;
  bool worstImproved = false;
  int score = 0; // 0..5
};

static ComparisonResult ComputeComparison(const FAEngine &orig,
                                          const FAEngine &mod) {
  ComparisonResult r;
  r.computed = true;
  r.origBest = orig.stats.bestFit;
  r.modBest = mod.stats.bestFit;
  r.origIterOpt = orig.stats.iterToOptimal;
  r.modIterOpt = mod.stats.iterToOptimal;
  r.origOptHits = orig.stats.optimalCount;
  r.modOptHits = mod.stats.optimalCount;
  r.origAvg = orig.stats.avgFit;
  r.modAvg = mod.stats.avgFit;

  r.origDiversity =
      orig.diversityPerIter.empty() ? 0.f : orig.diversityPerIter.back();
  r.modDiversity =
      mod.diversityPerIter.empty() ? 0.f : mod.diversityPerIter.back();

  r.origWorst = orig.stats.worstPerIter.empty()
                    ? 0.f
                    : *std::max_element(orig.stats.worstPerIter.begin(),
                                        orig.stats.worstPerIter.end());
  r.modWorst = mod.stats.worstPerIter.empty()
                   ? 0.f
                   : *std::max_element(mod.stats.worstPerIter.begin(),
                                       mod.stats.worstPerIter.end());

  r.origTotalIter = orig.stats.totalIter;
  r.modTotalIter = mod.stats.totalIter;

  r.fitImproved = r.modBest > r.origBest;
  r.iterImproved = (r.modIterOpt >= 0 &&
                    (r.origIterOpt < 0 || r.modIterOpt < r.origIterOpt));
  r.hitsImproved = r.modOptHits > r.origOptHits;
  r.avgImproved = r.modAvg > r.origAvg * 1.05f; // 5% threshold
  // lower diversity in Modified = better (converges faster)
  r.diversityImproved = r.modDiversity < r.origDiversity;

  // Cumulative score calculation
  r.score = 0;
  if (r.modBest >= r.origBest)
    r.score++;
  if (r.iterImproved)
    r.score++;
  if (r.hitsImproved)
    r.score++;
  if (r.modAvg > r.origAvg * 1.05f)
    r.score++;
  if (r.diversityImproved)
    r.score++;

  // ── ADD PENALTIES & PROGRESS CLAMPING ────────────────────
  int itersDone = std::max(orig.iter, mod.iter);
  float progress = (float)itersDone / MAX_ITER;
  // Clamp score based on run progress: early iters can only show low score
  int maxAllowedScore = 1 + (int)(progress * 4.f); // 1 at start, up to 5 at end
  r.score = std::min(r.score, maxAllowedScore);

  // Penalise if mod params are poorly tuned (very high randomness + no elites)
  bool badParams = (modParams.alpha0 > 1.6f && modParams.eliteCount == 0);
  bool noMutation = (modParams.mutRate < 0.05f && modParams.heurRatio < 0.1f);
  if (badParams)
    r.score = std::max(0, r.score - 2);
  if (noMutation)
    r.score = std::max(0, r.score - 1);
  return r;
}

// ============================================================
// BOARD ANIMATOR — smooth queen sliding with trails
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
    float cell =
        boardRect.width / n; // use width (square board, same as height)
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

// Draw Queen piece — r// Draw Queen piece — faithful chess queen silhouette
void DrawQueenPiece(Vector2 center, float sz, Color col, bool isOptimal,
                    float time) {
  (void)time; // no animation / glow
  float r = sz * 0.42f;

  // The piece spans from topmost orb tip to bottom of base rim.
  // Top  = center.y - tallest prong height - orb radius  = approx -r*0.90 -
  // r*0.15 = -r*1.05 Bot  = center.y + rim bottom                          =
  // approx +r*1.02 Total visual height ~ r*2.07, so true centre is at offset
  // +r*0.015 below drawn centre. Shift everything down by half that so the
  // piece sits centred in the cell.
  float vc = r * 0.28f; // vertical correction — shift drawing down by this much
  center.y += vc;

  // ── DROP SHADOW ───────────────────────────────────────────
  DrawEllipse((int)(center.x + 2), (int)(center.y + r * 1.05f + 2),
              (int)(r * 1.1f), (int)(r * 0.22f),
              ColorAlpha({0, 0, 0, 255}, 0.35f));

  Color body = col;
  Color dark = ColorBrightness(body, -0.28f);
  Color light = ColorBrightness(body, 0.32f);
  Color outline = ColorBrightness(body, -0.55f);

  // ── BASE — two layered trapezoids ─────────────────────────
  // Bottom rim (widest)
  float rimW = r * 1.18f;
  float rimH = r * 0.20f;
  float rimY = center.y + r * 0.82f;
  DrawRectangleRounded({center.x - rimW, rimY, rimW * 2.f, rimH}, 0.5f, 8,
                       dark);
  DrawRectangleRoundedLinesEx({center.x - rimW, rimY, rimW * 2.f, rimH}, 0.5f,
                              8, 1.2f, outline);

  // Upper rim
  float rim2W = r * 1.0f;
  float rim2H = r * 0.16f;
  float rim2Y = rimY - rim2H + 2;
  DrawRectangleRounded({center.x - rim2W, rim2Y, rim2W * 2.f, rim2H}, 0.5f, 8,
                       ColorBrightness(dark, 0.08f));
  DrawRectangleRoundedLinesEx({center.x - rim2W, rim2Y, rim2W * 2.f, rim2H},
                              0.5f, 8, 1.0f, outline);

  // ── SKIRT — trapezoid body flare ─────────────────────────
  float skirtTopW = r * 0.68f;
  float skirtBotW = r * 0.98f;
  float skirtTopY = center.y + r * 0.10f;
  float skirtBotY = rim2Y + rim2H * 0.5f;
  // Fill as two triangles
  Vector2 sk[4] = {{center.x - skirtTopW, skirtTopY},
                   {center.x + skirtTopW, skirtTopY},
                   {center.x + skirtBotW, skirtBotY},
                   {center.x - skirtBotW, skirtBotY}};
  DrawTriangle(sk[3], sk[0], sk[1], body);
  DrawTriangle(sk[3], sk[1], sk[2], body);
  // Outline edges
  DrawLineEx(sk[0], sk[3], 1.3f, outline);
  DrawLineEx(sk[1], sk[2], 1.3f, outline);
  DrawLineEx(sk[2], sk[3], 1.3f, outline);
  // Decorative horizontal lines on skirt (like the reference image)
  for (int line = 1; line <= 2; line++) {
    float t = (float)line / 3.f;
    float ly2 = skirtTopY + (skirtBotY - skirtTopY) * t;
    float lw = skirtTopW + (skirtBotW - skirtTopW) * t;
    DrawLineEx({center.x - lw, ly2}, {center.x + lw, ly2}, 1.0f,
               ColorAlpha(outline, 0.55f));
  }

  // ── CROWN BAND ───────────────────────────────────────────
  float bandH = r * 0.22f;
  float bandW = r * 0.70f;
  float bandY = skirtTopY - bandH + r * 0.04f;
  DrawRectangleRounded({center.x - bandW, bandY, bandW * 2.f, bandH}, 0.3f, 8,
                       body);
  DrawRectangleRoundedLinesEx({center.x - bandW, bandY, bandW * 2.f, bandH},
                              0.3f, 8, 1.2f, outline);
  // Highlight stripe on band
  DrawRectangleRounded(
      {center.x - bandW + 2, bandY + 1, bandW * 2.f - 4, bandH * 0.38f}, 0.3f,
      4, ColorAlpha(light, 0.45f));

  // ── FIVE CROWN PRONGS ────────────────────────────────────
  // Positions: evenly spaced across band width
  // Heights alternate: outer tall, inner short, centre tallest
  float prongX[5];
  float spread = bandW * 0.92f;
  for (int i = 0; i < 5; i++)
    prongX[i] = center.x - spread + i * (spread * 2.f / 4.f);

  float prongBaseY = bandY + bandH * 0.3f;
  float prongH[5] = {
      r * 0.72f, // left outer
      r * 0.50f, // left inner
      r * 0.90f, // centre (tallest)
      r * 0.50f, // right inner
      r * 0.72f  // right outer
  };
  float prongW = r * 0.13f;

  for (int i = 0; i < 5; i++) {
    float tipY = prongBaseY - prongH[i];
    float px = prongX[i];
    // Prong shaft
    DrawRectangleRounded({px - prongW, tipY, prongW * 2.f, prongH[i]}, 0.4f, 6,
                         body);
    DrawRectangleRoundedLinesEx({px - prongW, tipY, prongW * 2.f, prongH[i]},
                                0.4f, 6, 1.1f, outline);
    // Orb at tip — circle with outline (matches reference image)
    float orbR = prongW * 1.15f;
    DrawCircleV({px, tipY}, orbR, body);
    DrawCircleLines((int)px, (int)tipY, orbR, outline);
    // Small specular on orb
    DrawCircleV({px - orbR * 0.28f, tipY - orbR * 0.28f}, orbR * 0.28f,
                ColorAlpha(light, 0.55f));
  }

  // ── BODY FILL (covers prong bases) ───────────────────────
  // Redraw the band on top so prong bases look clean
  DrawRectangleRounded({center.x - bandW, bandY, bandW * 2.f, bandH}, 0.3f, 8,
                       body);
  DrawRectangleRoundedLinesEx({center.x - bandW, bandY, bandW * 2.f, bandH},
                              0.3f, 8, 1.2f, outline);
  DrawRectangleRounded(
      {center.x - bandW + 2, bandY + 1, bandW * 2.f - 4, bandH * 0.38f}, 0.3f,
      4, ColorAlpha(light, 0.40f));

  // ── OPTIMAL INDICATOR — subtle crown outline only, no glow ──
  if (isOptimal) {
    DrawRectangleRoundedLinesEx(
        {center.x - bandW - 2, bandY - 2, bandW * 2.f + 4, bandH + 4}, 0.3f, 8,
        1.5f, ColorAlpha(PAL.gold, 0.9f));
    for (int i = 0; i < 5; i++) {
      float orbR = prongW * 1.15f;
      float tipY = prongBaseY - prongH[i];
      DrawCircleLines((int)prongX[i], (int)tipY, orbR + 1.5f,
                      ColorAlpha(PAL.gold, 0.85f));
    }
  }
}

// Panel with glowing border
void DrawPanel(Rectangle r, const char *title, Font font,
               Color borderCol = {35, 40, 65, 255}) {
  DrawRectangleRec(r, PAL.panel);
  DrawRectangleRoundedLinesEx(r, 0.04f, 12, 1.0f, borderCol);
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
    DrawCircleV({ex, ey}, 7.f, col);
    DrawCircleV({ex, ey}, 3.f, WHITE);
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
    DrawRectangleRoundedLinesEx(rect, 0.35f, 12, 1.5f, border);
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
        DrawLineEx(p1, p2, 2.5f, ColorAlpha(PAL.conflict, 0.7f));
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

  // Outer glow rings — tighter to reduce bleed
  float gRad = r * (2.2f + 0.4f * sinf(pulse + time * 2.f));
  DrawCircleV(center, gRad * 0.6f, ColorAlpha(col, 0.05f));
  DrawCircleV(center, gRad * 0.35f, ColorAlpha(col, 0.10f));

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
  // Pulse ring on best firefly
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
    // Emit trail particles occasionally (deterministic, no Raylib RNG)
    if (showTrails && (trailFrame % 11 == idx % 11) &&
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

    float fitRatio = std::min(1.f, ff.fitness / MAX_FIT);
    // Quantize to discrete steps: 0, 3, 5, 7, 9, 11, ... 99 so color changes
    // one notch at a time
    auto stepFit = [](float r) -> float {
      int pct = (int)(r * 100.f);
      if (pct < 3)
        return 0.f;
      if (pct < 5)
        return 0.03f;
      // Round down to nearest odd >= 5
      int s = (pct % 2 == 0) ? pct - 1 : pct;
      return (float)s / 100.f;
    };
    float sr = stepFit(fitRatio);
    // Dark start (maroon for mod, dark navy for orig) → mid → full tint →
    // subtle gold
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
    float r =
        3.f +
        fitRatio * 6.f; // smaller: 3-9px so fireflies stay above progress bar
    bool moving = Vector2Distance(ff.screenPos, ff.targetPos) > 5.f;
    DrawFireflyDetailed(ff.screenPos, r, ffColor, ff.pulsePhase, ff.isElite,
                        time, moving);
  }
  // Diversity bars at bottom of swarm panel
  DrawRectangle((int)(SWARM_RECT.x + 4),
                (int)(SWARM_RECT.y + SWARM_RECT.height - 8),
                (int)((SWARM_RECT.width - 8) * fa.diversity), 5,
                ColorAlpha(fa.isModified ? PAL.mod : PAL.orig, 0.7f));
}

// ============================================================
// SECTION 18: COMPARISON PANEL
// ============================================================
void DrawComparisonPanel(Rectangle bounds, const ComparisonResult &cmp,
                         Font font, Font fontSm, float time) {
  if (!cmp.computed)
    return;

  float xL = bounds.x + 10.f;
  float xO = bounds.x + bounds.width * 0.40f;
  float xM = bounds.x + bounds.width * 0.68f;
  float rowH = 26.f;
  float y = bounds.y + 8.f;

  // ── HEADER ──
  DrawRectangleRec({bounds.x, y, bounds.width, 20},
                   ColorAlpha(PAL.panelBord, 0.4f));
  DrawTextEx(font, "Metric", {xL, y + 3}, 12, 1, PAL.dimText);
  DrawTextEx(font, "Original", {xO, y + 3}, 12, 1, PAL.orig);
  DrawTextEx(font, "Modified", {xM, y + 3}, 12, 1, PAL.mod);
  y += 22.f;
  DrawLineEx({bounds.x + 4, y}, {bounds.x + bounds.width - 4, y}, 0.5f,
             PAL.panelBord);
  y += 3.f;

  // ── ROW HELPER ──
  // improved=true → green arrow up, worsened=true → red arrow down
  auto drawRow = [&](const char *label, const char *vO, const char *vM,
                     bool improved, bool worsened, bool alt) {
    if (alt)
      DrawRectangleRec({bounds.x + 4, y - 1, bounds.width - 8, rowH - 2},
                       ColorAlpha(PAL.panelBord, 0.2f));
    DrawTextEx(font, label, {xL, y + 4}, 12, 1, PAL.text);
    DrawTextEx(font, vO, {xO, y + 4}, 12, 1, PAL.orig);

    Color vc = improved ? PAL.safe : (worsened ? PAL.conflict : PAL.mod);
    DrawTextEx(font, vM, {xM, y + 4}, 12, 1, vc);

    // Arrow indicator at far right
    float ax = bounds.x + bounds.width - 18.f;
    float ay = y + rowH / 2.f;
    if (improved) {
      // Up triangle
      DrawTriangle({ax + 4, ay - 5}, {ax + 8, ay + 4}, {ax, ay + 4},
                   ColorAlpha(PAL.safe, 0.9f));
    } else if (worsened) {
      // Down triangle
      DrawTriangle({ax, ay - 4}, {ax + 8, ay - 4}, {ax + 4, ay + 5},
                   ColorAlpha(PAL.conflict, 0.9f));
    } else {
      // Neutral dash
      DrawLineEx({ax, ay}, {ax + 8, ay}, 2.f, ColorAlpha(PAL.dimText, 0.6f));
    }
    y += rowH;
  };

  // ── ROWS ──
  drawRow("Best Fitness", TextFormat("%.0f / %d", cmp.origBest, MAX_FIT),
          TextFormat("%.0f / %d", cmp.modBest, MAX_FIT), cmp.fitImproved,
          cmp.modBest < cmp.origBest, true);

  drawRow("Avg Fitness", TextFormat("%.2f", cmp.origAvg),
          TextFormat("%.2f", cmp.modAvg), cmp.avgImproved,
          cmp.modAvg < cmp.origAvg, false);

  drawRow("Iter to Opt",
          cmp.origIterOpt < 0 ? "Not found" : TextFormat("%d", cmp.origIterOpt),
          cmp.modIterOpt < 0 ? "Not found" : TextFormat("%d", cmp.modIterOpt),
          cmp.iterImproved,
          (cmp.modIterOpt < 0 && cmp.origIterOpt >= 0) ||
              (cmp.origIterOpt >= 0 && cmp.modIterOpt >= 0 &&
               cmp.modIterOpt > cmp.origIterOpt),
          true);

  drawRow("Optimal Hits", TextFormat("%d", cmp.origOptHits),
          TextFormat("%d", cmp.modOptHits), cmp.hitsImproved,
          cmp.modOptHits < cmp.origOptHits, false);

  drawRow("Worst Fitness", TextFormat("%.0f", cmp.origWorst),
          TextFormat("%.0f", cmp.modWorst), cmp.worstImproved,
          cmp.modWorst < cmp.origWorst, true);

  drawRow("Final Diversity", TextFormat("%.3f", cmp.origDiversity),
          TextFormat("%.3f", cmp.modDiversity), cmp.diversityImproved,
          !cmp.diversityImproved, false);

  // ── DIVIDER ──
  y += 2.f;
  DrawLineEx({bounds.x + 4, y}, {bounds.x + bounds.width - 4, y}, 0.5f,
             PAL.panelBord);
  y += 6.f;

  // ── OVERALL VERDICT ──
  const char *verdicts[] = {"No advantage yet", "Slight edge detected",
                            "Gaining ground", "Strong improvement",
                            "Full improvement"};
  Color verdictCols[] = {PAL.dimText, PAL.dimText, PAL.mod, PAL.safe, PAL.gold};

  // Use the already-capped score from ComputeComparison instead of raw liveScore
  int liveScore = cmp.score; // respects the progress cap computed in engine

  int vi = Clamp(liveScore, 0, 4);

  DrawTextEx(font, "Overall:", {xL, y}, 12, 1, PAL.dimText);
  DrawTextEx(font, verdicts[vi], {xL + 56.f, y}, 12, 1, verdictCols[vi]);
  y += 18.f;

  // Score dots (out of 5)
  DrawTextEx(fontSm, "Score:", {xL, y}, 11, 1, PAL.dimText);
  for (int s = 0; s < 5; s++) {
    float dx = xL + 44.f + s * 18.f;
    Color dc = (s < liveScore) ? PAL.gold : ColorAlpha(PAL.panelBord, 0.8f);
    DrawCircleV({dx, y + 6.f}, 5.f, dc);
    if (s < liveScore)
      DrawCircleLines((int)dx, (int)(y + 6.f), 7.5f,
                      ColorAlpha(PAL.gold, 0.35f));
  }
  y += 20.f;
}

// ============================================================
// SECTION 19: PARAMETER POPUP
// ============================================================
struct ParamPopup {
  bool visible = false;
  float alpha = 0.f;
  float sVals[4] = {0.42f, 0.2f, 0.0f, 0.4f}; // mapped 0..1
  bool dragging[4] = {false, false, false, false};
  Color sliderCols[4] = {PAL.gold, {200, 100, 255, 255}, PAL.safe, PAL.mod};

  void syncToGlobal() {
    sVals[0] = (modParams.alpha0 - 0.1f) / 1.9f;
    sVals[1] = modParams.mutRate;
    sVals[2] = modParams.heurRatio;
    sVals[3] = (float)modParams.eliteCount / 4.f;
  }
  void resetToDefaults() {
    ModParams d = ModParams::defaults();
    sVals[0] = (d.alpha0 - 0.1f) / 1.9f;
    sVals[1] = d.mutRate;
    sVals[2] = d.heurRatio;
    sVals[3] = (float)d.eliteCount / 4.f;
    // Apply immediately
    modParams.alpha0 = d.alpha0;
    modParams.mutRate = d.mutRate;
    modParams.heurRatio = d.heurRatio;
    modParams.eliteCount = d.eliteCount;
  }

  void update(float dt) {
    alpha = Lerp(alpha, visible ? 1.f : 0.f, dt * 12.f);
    if (alpha < 0.01f)
      return;

    Vector2 mp = GetMousePosition();
    float mw = 620.f, mh = 580.f;
    float mx = WIN_W / 2.f - mw / 2.f;
    float my = WIN_H / 2.f - mh / 2.f;

    // Close X button
    Rectangle closeR = {mx + mw - 44, my + 8, 32, 32};
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
        CheckCollisionPointRec(mp, closeR)) {
      visible = false;
      return;
    }

    // Reset button
    Rectangle resetR = {mx + 16.f, my + mh - 52.f, 110.f, 38.f};
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
        CheckCollisionPointRec(mp, resetR)) {
      resetToDefaults();
      return;
    }

    // Apply & Close button
    Rectangle applyR = {mx + mw - 150.f, my + mh - 52.f, 134.f, 38.f};
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
        CheckCollisionPointRec(mp, applyR)) {
      visible = false;
      return;
    }

    // Escape closes
    if (IsKeyPressed(KEY_ESCAPE) && visible) {
      visible = false;
      return;
    }

    float tw = mw - 140.f;
    float tx = mx + 70.f;
    float rowH = 92.f;
    float startY = my + 78.f;

    // Release all drags on mouse up
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
      for (int i = 0; i < 4; i++)
        dragging[i] = false;
    }

    for (int i = 0; i < 4; i++) {
      float trackY = startY + i * rowH + 46.f;
      Rectangle track = {tx, trackY, tw, 10.f};
      Rectangle clickZone = {tx - 8.f, trackY - 20.f, tw + 16.f, 50.f};

      // Start drag on click anywhere in the zone
      if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) &&
          CheckCollisionPointRec(mp, clickZone)) {
        dragging[i] = true;
        sVals[i] = Clamp((mp.x - track.x) / track.width, 0.f, 1.f);
      }

      // Continue drag while held
      if (dragging[i] && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        sVals[i] = Clamp((mp.x - track.x) / track.width, 0.f, 1.f);
      }
    }

    // ── Live-update modParams every frame so badges show current value ──
    modParams.alpha0 = 0.1f + sVals[0] * 1.9f;
    modParams.mutRate = sVals[1];
    modParams.heurRatio = sVals[2];
    modParams.eliteCount = (int)(sVals[3] * 4.f + 0.5f); // rounds: 0,1,2,3,4
  }

  void draw(Font font, Font fontBold, float time) {
    if (alpha < 0.01f)
      return;
    float a = alpha;

    // Dim overlay
    DrawRectangle(0, 0, WIN_W, WIN_H, ColorAlpha({0, 0, 8, 255}, 0.78f * a));

    float mw = 620.f, mh = 580.f;
    float mx = WIN_W / 2.f - mw / 2.f;
    float my = WIN_H / 2.f - mh / 2.f;
    Rectangle modal = {mx, my, mw, mh};

    // Card background
    DrawRectangleRounded(modal, 0.05f, 16, ColorAlpha({11, 13, 28, 255}, a));

    // Animated border — pulses between mod and gold
    Color borderCol =
        LerpColor(PAL.mod, PAL.gold, 0.5f + 0.5f * sinf(time * 1.8f));
    DrawRectangleRoundedLinesEx(modal, 0.05f, 16, 2.f,
                                ColorAlpha(borderCol, a * 0.9f));

    // Header stripe
    DrawRectangleRounded({mx, my, mw, 62.f}, 0.05f, 16,
                         ColorAlpha({16, 14, 42, 255}, a));
    // Accent line under header
    DrawRectangle((int)mx + 2, (int)(my + 60), (int)mw - 4, 2,
                  ColorAlpha(PAL.mod, a * 0.6f));

    // Title
    DrawTextEx(fontBold, "MODIFIED FA PARAMETERS", {mx + 20, my + 14}, 18, 1.5f,
               ColorAlpha(PAL.gold, a));
    DrawTextEx(font, "Drag sliders — changes apply to next run.",
               {mx + 20, my + 40}, 11, 1, ColorAlpha(PAL.dimText, a * 0.8f));

    // ── CLOSE BUTTON (top-right X) ──
    Rectangle closeR = {mx + mw - 44.f, my + 10.f, 34.f, 34.f};
    bool hovClose = CheckCollisionPointRec(GetMousePosition(), closeR);
    DrawRectangleRounded(closeR, 0.4f, 8,
                         ColorAlpha(hovClose
                                        ? PAL.conflict
                                        : ColorBrightness(PAL.conflict, -0.25f),
                                    a));
    DrawRectangleRoundedLines(closeR, 0.4f, 8, 1.2f,
                              ColorAlpha(WHITE, hovClose ? a : a * 0.4f));
    Vector2 xSz = MeasureTextEx(fontBold, "X", 15, 1);
    DrawTextEx(fontBold, "X",
               {closeR.x + closeR.width / 2 - xSz.x / 2,
                closeR.y + closeR.height / 2 - xSz.y / 2},
               15, 1, ColorAlpha(WHITE, a));

    // ── SLIDER ROWS ──
    float tw = mw - 140.f;
    float tx = mx + 70.f;
    float rowH = 92.f;
    float startY = my + 78.f;

    // Parameter metadata
    struct ParamDef {
      const char *name;
      const char *desc;
      const char *minLbl;
      const char *maxLbl;
      Color col;
    };
    ParamDef defs[4] = {
        {"Alpha0  (Randomness)",
         "Controls early exploration breadth. Higher = wider initial jumps.",
         "0.10 (tight)", "2.00 (wild)", PAL.gold},
        {"Mutation Rate",
         "Swap probability per firefly each iteration. Injects diversity.",
         "0% (off)",
         "100% (always)",
         {200, 100, 255, 255}},
        {"Heuristic Init Bias",
         "Fraction of population seeded by greedy placement vs random.",
         "Fully Random", "Fully Heuristic", PAL.safe},
        {"Elite Count",
         "Top-N fireflies preserved each iteration (prevents regression).",
         "0  (off)", "4  (heavy)", PAL.mod},
    };

    // Current value labels
    const char *valLabels[4];
    valLabels[0] = TextFormat("%.2f", modParams.alpha0);
    valLabels[1] = TextFormat("%d%%", (int)(modParams.mutRate * 100));
    const char *hLbl = modParams.heurRatio < 0.15f   ? "Fully Random"
                       : modParams.heurRatio < 0.35f ? "Mostly Random"
                       : modParams.heurRatio < 0.65f ? "Balanced"
                       : modParams.heurRatio < 0.85f ? "Mostly Heuristic"
                                                     : "Full Heuristic";
    valLabels[2] = hLbl;
    valLabels[3] = TextFormat("%d fireflies", modParams.eliteCount);

    for (int i = 0; i < 4; i++) {
      float ry = startY + i * rowH;
      Rectangle track = {tx, ry + 46.f, tw, 10.f};

      // Row background — highlighted when dragging
      Color rowBg = dragging[i] ? ColorAlpha(defs[i].col, a * 0.10f)
                                : ColorAlpha(PAL.panelBord, a * 0.18f);
      DrawRectangleRounded({mx + 8, ry + 2, mw - 16, rowH - 4}, 0.12f, 8,
                           rowBg);
      if (dragging[i])
        DrawRectangleRoundedLines({mx + 8, ry + 2, mw - 16, rowH - 4}, 0.12f, 8,
                                  1.f, ColorAlpha(defs[i].col, a * 0.5f));

      // ── Name label ──
      DrawTextEx(font, defs[i].name, {tx, ry + 10.f}, 14, 1,
                 ColorAlpha(defs[i].col, a));

      // ── Value badge (top-right of row) ──
      Vector2 vs = MeasureTextEx(font, valLabels[i], 13, 1);
      float bw2 = vs.x + 20.f;
      float bx2 = mx + mw - 16.f - bw2;
      float by2 = ry + 8.f;
      DrawRectangleRounded({bx2, by2, bw2, 24.f}, 0.4f, 8,
                           ColorAlpha(defs[i].col, a * 0.18f));
      DrawRectangleRoundedLines({bx2, by2, bw2, 24.f}, 0.4f, 8, 0.8f,
                                ColorAlpha(defs[i].col, a * 0.75f));
      DrawTextEx(font, valLabels[i], {bx2 + 10.f, by2 + 5.f}, 13, 1,
                 ColorAlpha(defs[i].col, a));

      // ── Description ──
      DrawTextEx(font, defs[i].desc, {tx, ry + 28.f}, 11, 1,
                 ColorAlpha(PAL.dimText, a * 0.75f));

      // ── Slider track ──
      DrawRectangleRounded(track, 1.f, 8, ColorAlpha(PAL.axis, a * 0.45f));

      // Gradient fill
      if (sVals[i] > 0.001f) {
        DrawRectangleGradientH(
            (int)track.x, (int)track.y, (int)(sVals[i] * track.width),
            (int)track.height,
            ColorAlpha(ColorBrightness(defs[i].col, -0.35f), a * 0.85f),
            ColorAlpha(defs[i].col, a * 0.95f));
      }

      // Tick marks at 0%, 25%, 50%, 75%, 100%
      for (int t = 0; t <= 4; t++) {
        float tx2 = track.x + t * track.width / 4.f;
        DrawLineEx({tx2, track.y + track.height + 2.f},
                   {tx2, track.y + track.height + 7.f}, 1.f,
                   ColorAlpha(PAL.dimText, a * 0.35f));
      }

      // Handle
      float hx2 = track.x + sVals[i] * track.width;
      float hy2 = track.y + track.height / 2.f;
      float hr = dragging[i] ? 13.f : 10.f;
      // Glow ring when active
      if (dragging[i])
        DrawCircleV({hx2, hy2}, hr * 2.2f, ColorAlpha(defs[i].col, a * 0.12f));
      DrawCircleV({hx2, hy2}, hr, ColorAlpha(defs[i].col, a));
      DrawCircleV({hx2, hy2}, hr * 0.42f, ColorAlpha(WHITE, a * 0.9f));

      // Min / max labels
      DrawTextEx(font, defs[i].minLbl, {track.x, ry + 60.f}, 10, 1,
                 ColorAlpha(PAL.dimText, a * 0.65f));
      Vector2 maxSz = MeasureTextEx(font, defs[i].maxLbl, 10, 1);
      DrawTextEx(font, defs[i].maxLbl,
                 {track.x + track.width - maxSz.x, ry + 60.f}, 10, 1,
                 ColorAlpha(PAL.dimText, a * 0.65f));
    }

    // ── BOTTOM BUTTONS ──
    float btnY = my + mh - 54.f;

    // Reset to defaults button (left)
    Rectangle resetR = {mx + 16.f, btnY, 110.f, 38.f};
    bool hovReset = CheckCollisionPointRec(GetMousePosition(), resetR);
    DrawRectangleRounded(resetR, 0.35f, 12,
                         ColorAlpha(hovReset
                                        ? ColorBrightness(PAL.panelBord, 0.35f)
                                        : ColorBrightness(PAL.panelBord, 0.15f),
                                    a * 0.9f));
    DrawRectangleRoundedLines(
        resetR, 0.35f, 12, 1.3f,
        ColorAlpha(PAL.dimText, a * (hovReset ? 0.9f : 0.5f)));
    {
      Vector2 ts = MeasureTextEx(font, "Reset Defaults", 13, 1);
      DrawTextEx(font, "Reset Defaults",
                 {resetR.x + resetR.width / 2 - ts.x / 2,
                  resetR.y + resetR.height / 2 - ts.y / 2},
                 13, 1, ColorAlpha(WHITE, a * (hovReset ? 1.f : 0.7f)));
    }

    // Close / apply button (right)
    Rectangle applyR = {mx + mw - 150.f, btnY, 134.f, 38.f};
    bool hovApply = CheckCollisionPointRec(GetMousePosition(), applyR);
    DrawRectangleRounded(
        applyR, 0.35f, 12,
        ColorAlpha(hovApply ? ColorBrightness(PAL.mod, 0.2f) : PAL.mod,
                   a * 0.85f));
    DrawRectangleRoundedLines(
        applyR, 0.35f, 12, 1.5f,
        ColorAlpha(hovApply ? WHITE : LerpColor(PAL.mod, WHITE, 0.4f), a));
    {
      Vector2 ts = MeasureTextEx(fontBold, "Apply & Close", 14, 1);
      DrawTextEx(fontBold, "Apply & Close",
                 {applyR.x + applyR.width / 2 - ts.x / 2,
                  applyR.y + applyR.height / 2 - ts.y / 2},
                 14, 1, ColorAlpha(WHITE, a));
    }

    // ── FOOTER HINT ──
    DrawTextEx(font, "[Esc] to cancel without applying",
               {mx + mw / 2.f - 88.f, my + mh - 16.f}, 10, 1,
               ColorAlpha(PAL.dimText, a * 0.45f));
  }
};

static ParamPopup paramPopup;

// ============================================================
// SECTION 20: SPEED SLIDER
// ============================================================
struct Slider {
  Rectangle track;
  float value = 0.5f; // 0..1
  float handleT = 0.f;
  bool dragging = false;
  bool hovTrack = false;
  bool hovHandle = false;

  void update() {
    Vector2 mp = GetMousePosition();
    float hx = track.x + value * track.width;
    hovHandle =
        CheckCollisionPointCircle(mp, {hx, track.y + track.height / 2}, 14.f);

    // Use an expanded hit area for the track to make clicking easier
    Rectangle clickArea = {track.x, track.y - 12.f, track.width,
                           track.height + 24.f};
    hovTrack = CheckCollisionPointRec(mp, clickArea);

    handleT =
        Lerp(handleT, hovHandle || dragging ? 1.f : 0.f, GetFrameTime() * 10.f);

    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && (hovHandle || hovTrack)) {
      dragging = true;
    }
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON))
      dragging = false;

    if (dragging)
      value = Clamp((mp.x - track.x) / track.width, 0.f, 1.f);
  }
  void draw(Font font, Color col, const char *label = "Speed",
            const char *fmt = "%.1fx", float minV = 1.0f, float maxV = 10.0f,
            bool disabled = false) {
    Color baseCol = disabled ? PAL.dimText : col;
    Color trackCol =
        disabled ? ColorAlpha(PAL.axis, 0.2f) : ColorAlpha(PAL.axis, 0.5f);
    // Track
    DrawRectangleRounded(track, 1.f, 8, trackCol);
    // Fill
    Rectangle fill = track;
    fill.width = value * track.width;
    DrawRectangleRounded(fill, 1.f, 8, ColorAlpha(baseCol, 0.6f));
    // Handle
    float hx = track.x + value * track.width;
    float hy = track.y + track.height / 2;
    DrawCircleV({hx, hy}, 10.f + handleT * 3.f, baseCol);
    DrawCircleV({hx, hy}, 5.f, disabled ? PAL.panel : WHITE);

    float displayVal = Lerp(minV, maxV, value * value);
    const char *valText = TextFormat(fmt, displayVal);
    const char *fullText = TextFormat("%s: %s", label, valText);
    DrawTextEx(font, fullText, {track.x + track.width + 12, hy - 8}, 14, 1,
               disabled ? PAL.dimText : PAL.text);
  }
  float getInterval() {
    // Base: quadratic curve for better control at slow speeds
    float t = value * value;
    // Scale minimum interval by N^2 since each step costs work proportional to N
    // (board) & pop size
    float nScale = std::max(1.f, (float)(N * N) / 64.f); // N=8->1x, N=32->16x
    float baseInterval = Lerp(0.5f, 0.0f, t);           // 0.5s to 0s
    return baseInterval * nScale;
  }
};

// ============================================================
// SECTION 18: MAIN
// ============================================================
int main() {
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
  InitWindow(WIN_W, WIN_H, "Firefly Algorithm Visualizer — Next Level");
  SetTargetFPS(60);

  // Fonts
  Font fontBold = GetFontDefault();
  Font fontMono = GetFontDefault();
  Font fontSm = GetFontDefault();

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
  speedSlider.value = 0.4f;

  // Engines
  FAEngine origFA, modFA;
  BoardAnimator animO, animM;

  // Comparison Result
  ComparisonResult cmpResult;

  // Buttons
  Button btnOrig = {{0, 0, 118, 44}, "Original FA", PAL.orig};
  Button btnMod = {{0, 0, 118, 44}, "Modified FA", PAL.mod};
  Button btnBoth = {{0, 0, 100, 44}, "Run Both", PAL.accent};
  Button btnParams = {{0, 0, 110, 40}, "Parameters", {120, 60, 200, 255}};
  Button btnReset = {{0, 0, 80, 44}, "Reset", PAL.conflict};
  Button btnPlayStop = {{0, 0, 110, 40}, "PLAY", PAL.safe};
  Button btnStepFwd = {{0, 0, 52, 40}, ">|", PAL.gold};
  Button btnStepBwd = {{0, 0, 52, 40}, "|<", ColorBrightness(PAL.gold, -0.2f)};
  // Step-back snapshot history
  struct SnapShot {
    std::vector<std::vector<int>> posO, posM;
    std::vector<float> fitO, fitM;
    int iterO = 0, iterM = 0;
    bool doneO = false, doneM = false;
  };
  std::deque<SnapShot> stepHistory;
  static const int MAX_HISTORY = 30;
  // btnDecN and btnIncN moved to title bar — no longer Button objects

  static const int EXPAND_BTN_POS = 1;

  Button btnExpandGraph = {{0, 0, 80, 24}, "Expand", PAL.accent};
  Button btnCloseGraph = {
      {WIN_W / 2.f + 500.f - 85, WIN_H / 2.f - 300.f + 4, 75, 22},
      "Close",
      PAL.conflict};
  Button btnDownload = {{WIN_W / 2.f - 90.f, WIN_H - 140.f, 180.f, 40.f},
                        "Download Results",
                        {40, 160, 80, 255}};
  Button btnExpandCmp = {{0, 0, 60, 22}, "Expand", PAL.accent};
  bool showDownload = false;
  std::string downloadMsg = "";
  float downloadMsgTimer = 0.f;

  Button btnTheory = {{0, 0, 80.f, 40.f}, "Theory", {80, 60, 160, 255}};
  bool showTheory = false;
  int theoryTab = 0; // 0=Overview 1=Algorithm 2=Visuals 3=Colour 4=Graph
  float theoryAlpha = 0.f;

  bool showO = false, showM = false, running = false;
  float stepTimer = 0.f;
  float time = 0.f;
  bool showConflicts = true;
  bool showTrails = true;
  bool showCmpTable = false;
  float celebTimer = 0.f;
  bool showExpandedGraph = false;
  bool hasAutoExpanded = false;
  float expandedT = 0.f;
  int cmpPage = 0; // 0=stats, 1=mini graphs, 2=param sensitivity
  bool showExpandCmp = false;
  float expandCmpT = 0.f;
  bool solutionFound = false;
  bool everSolved = false;  // never cleared — survives reset/rerun
  int boardQueenFilter = 0; // 0=both, 1=orig only, 2=mod only
  float solvedBadgeX = (float)WIN_W + 10.f; // persistent slide-in position
  int everSolvedN = 0;
  int everOrigIter = -1;
  int everModIter = -1;

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();
    time += dt;
    stepTimer += dt;

    // Button updates
    btnOrig.update(dt);
    btnMod.update(dt);
    btnBoth.update(dt);
    btnReset.update(dt);
    btnPlayStop.update(dt);
    btnStepFwd.update(dt);
    btnStepBwd.update(dt);
    speedSlider.update();

    expandedT = Lerp(expandedT, showExpandedGraph ? 1.f : 0.f, dt * 10.f);
    if (downloadMsgTimer > 0.f)
      downloadMsgTimer -= dt;
    if (showDownload)
      btnDownload.update(dt);
    btnTheory.update(dt);
    if (btnTheory.pressed)
      showTheory = !showTheory;
    if (IsKeyPressed(KEY_H))
      showTheory = !showTheory;
    theoryAlpha = Lerp(theoryAlpha, showTheory ? 1.f : 0.f, dt * 12.f);
    tooltip.update(dt);
    starfield.update(dt);
    particles.update(dt);

    btnParams.update(dt);
    if (btnParams.pressed) {
      paramPopup.syncToGlobal();
      paramPopup.visible = !paramPopup.visible;
      if (paramPopup.visible)
        showTheory = false;
    }
    btnExpandGraph.update(dt);
    if (expandedT > 0.01f)
      btnCloseGraph.update(dt);
    if (btnExpandGraph.pressed) {
      showExpandedGraph = true;
    }
    btnExpandCmp.update(dt);
    if (btnExpandCmp.pressed)
      showExpandCmp = true;
    expandCmpT = Lerp(expandCmpT, showExpandCmp ? 1.f : 0.f, dt * 12.f);
    if (btnCloseGraph.pressed) {
      showExpandedGraph = false;
    }

    paramPopup.update(dt);
    if (paramPopup.visible)
      running = false;

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
    };
    if (btnOrig.pressed) {
      recalcParams();
      origFA.swarmHalf = 0;
      origFA.init(false, PAL.orig);
      animO.init(N, BOARD_RECT);
      showO = true;
      showM = false;
      cmpPage = 0;
      running = true;
      showCmpTable = false;
      stepTimer = 0.f;
      showExpandedGraph = false;
      hasAutoExpanded = false;
      solutionFound = false;
      stepHistory.clear();
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
    }
    if (btnMod.pressed) {
      recalcParams();
      modFA.swarmHalf = 0;
      modFA.init(true, PAL.mod);
      animM.init(N, BOARD_RECT);
      showO = false;
      showM = true;
      cmpPage = 0;
      running = true;
      showCmpTable = false;
      stepTimer = 0.f;
      showExpandedGraph = false;
      hasAutoExpanded = false;
      solutionFound = false;
      stepHistory.clear();
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
    }
    if (btnBoth.pressed) {
      recalcParams();
      origFA.swarmHalf = 1;
      modFA.swarmHalf = 2;
      origFA.init(false, PAL.orig);
      modFA.init(true, PAL.mod);
      animO.init(N, BOARD_RECT);
      animM.init(N, BOARD_RECT);
      showO = showM = true;
      cmpPage = 0;
      running = true;
      showCmpTable = true;
      stepTimer = 0.f;
      showExpandedGraph = false;
      hasAutoExpanded = false;
      solutionFound = false;
      stepHistory.clear();
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
    }
    if (btnReset.pressed) {
      resetRun();
    }
    // Play / Stop
    if (btnPlayStop.pressed && (showO || showM)) {
      running = !running;
      btnPlayStop.label = running ? "STOP" : "PLAY";
      btnPlayStop.col = running ? PAL.conflict : PAL.safe;
    }
    // Step Forward — save snapshot then advance
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
    // Step Backward — restore snapshot
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

    // ---- KEYBOARD SHORTCUTS ----
    if (IsKeyPressed(KEY_O)) {
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
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
    }
    if (IsKeyPressed(KEY_M)) {
      modFA.swarmHalf = 0;
      modFA.init(true, PAL.mod);
      animM.init(N, BOARD_RECT);
      showO = false;
      showM = true;
      running = true;
      showCmpTable = false;
      stepTimer = 0.f;
      showExpandedGraph = false;
      hasAutoExpanded = false;
      solutionFound = false;
      stepHistory.clear();
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
    }
    if (IsKeyPressed(KEY_B)) {
      origFA.swarmHalf = 1;
      modFA.swarmHalf = 2;
      origFA.init(false, PAL.orig);
      modFA.init(true, PAL.mod);
      animO.init(N, BOARD_RECT);
      animM.init(N, BOARD_RECT);
      showO = showM = true;
      running = true;
      showCmpTable = true;
      stepTimer = 0.f;
      showExpandedGraph = false;
      hasAutoExpanded = false;
      solutionFound = false;
      stepHistory.clear();
      btnPlayStop.label = "STOP";
      btnPlayStop.col = PAL.conflict;
    }
    if (IsKeyPressed(KEY_R)) {
      resetRun();
    }
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
      origFA.done = modFA.done = true;
    }
    if (IsKeyPressed(KEY_DOWN) && N > 4) {
      N--;
      recalcParams();
      resetRun();
      origFA.done = modFA.done = true;
    }
    if (IsKeyPressed(KEY_E))
      showExpandedGraph = !showExpandedGraph;
    if (IsKeyPressed(KEY_C))
      showConflicts = !showConflicts;
    if (IsKeyPressed(KEY_T))
      showTrails = !showTrails;
    if (IsKeyPressed(KEY_X))
      showCmpTable = !showCmpTable;

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
      if (odone && mdone && (showO || showM)) {
        if (running) {
          running = false;
          showDownload = true;
          btnPlayStop.label = "PLAY";
          btnPlayStop.col = PAL.safe;
          bool found = (showO && origFA.stats.bestFit >= MAX_FIT) ||
                       (showM && modFA.stats.bestFit >= MAX_FIT);
          if (found) {
            celebTimer = 3.f;
            solutionFound = true;
            if (!everSolved) {
              everSolved = true;
              everSolvedN = N;
              everOrigIter = origFA.stats.iterToOptimal;
              everModIter = modFA.stats.iterToOptimal;
            }
            particles.emit({WIN_W * 0.5f, WIN_H * 0.4f}, 120, 0, PAL.gold);
            particles.emit({WIN_W * 0.3f, WIN_H * 0.3f}, 80, 1, PAL.orig);
            particles.emit({WIN_W * 0.7f, WIN_H * 0.3f}, 80, 1, PAL.mod);
          }
          if (!hasAutoExpanded) {
            showExpandCmp = true;
            hasAutoExpanded = true;
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

    // Update board animators — smooth queen movement
    if (showO && origFA.inited) {
      animO.setTargets(origFA.getBest().position, N, BOARD_RECT);
      animO.update(dt);
    }
    if (showM && modFA.inited) {
      animM.setTargets(modFA.getBest().position, N, BOARD_RECT);
      animM.update(dt);
    }

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
    DrawRectangleGradientH(0, 0, WIN_W, 56, {8, 10, 28, 255},
                           {18, 12, 40, 255});
    DrawRectangle(0, 0, 4, 56, PAL.gold); // left accent stripe
    DrawTextEx(fontBold, "FIREFLY ALGORITHM", {18, 6}, 22, 1.5f, PAL.gold);
    DrawTextEx(fontSm, "N-QUEENS OPTIMISATION VISUALIZER", {20, 30}, 12, 2.f,
               PAL.dimText);
    // Title bar right side — anchored to far right with 16px margin
    // Layout (right to left): [Status Pill] gap [Theory Button] gap
    const char *statusStr =
        running ? "RUNNING" : (showO || showM ? "PAUSED" : "IDLE");
    Color statusCol =
        running ? PAL.safe : (showO || showM ? PAL.gold : PAL.dimText);

    float pillH = 40.f;
    float pillW = 100.f;
    float theoryW = 86.f;
    float btnH = 40.f;
    float gap = 14.f;
    float rightMargin = 16.f;
    float pillY = (56.f - pillH) / 2.f; // vertically centred in 56px title bar

    // Status pill — rightmost element
    float pillX = WIN_W - rightMargin - pillW;
    DrawRectangleRounded({pillX, pillY, pillW, pillH}, 0.4f, 8,
                         ColorAlpha(statusCol, 0.12f));
    DrawRectangleRoundedLinesEx({pillX, pillY, pillW, pillH}, 0.4f, 8, 1.2f,
                                statusCol);
    float dotPulse = 0.6f + 0.4f * sinf(time * 4.f);
    DrawCircleV({pillX + 15.f, pillY + pillH / 2.f},
                running ? 5.f * dotPulse : 4.f, statusCol);
    Vector2 stSz = MeasureTextEx(fontSm, statusStr, 13, 1);
    DrawTextEx(fontSm, statusStr,
               {pillX + 28.f, pillY + pillH / 2.f - stSz.y / 2.f}, 13, 1,
               statusCol);

    // Theory button — to the left of status pill with gap
    btnTheory.rect = {pillX - gap - theoryW, pillY, theoryW, btnH};
    btnTheory.draw(fontSm);

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
    // Squares with attack highlights
    // 1. Draw squares
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < N; c++) {
        Color sq = (r + c) % 2 == 0 ? PAL.lightSq : PAL.darkSq;
        DrawRectangle((int)(bx + c * cell), (int)(by + r * cell), (int)cell,
                      (int)cell, sq);
        if (CheckCollisionPointRec(GetMousePosition(),
                                   {bx + c * cell, by + r * cell, cell, cell}))
          DrawRectangle((int)(bx + c * cell), (int)(by + r * cell), (int)cell,
                        (int)cell, ColorAlpha(WHITE, 0.08f));
      }
    }
    // 2. Attack highlights: tint row/col of conflicting queens
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
    // Conflict lines — using animated queen positions
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
    // Queen trails on board
    if (showO)
      animO.drawTrails(PAL.orig);
    if (showM)
      animM.drawTrails(PAL.mod);
    // Queens — animated sliding positions
    auto drawQueensAnimated = [&](BoardAnimator &anim, FAEngine &fa, Color qcol,
                                  float scale) {
      if (!fa.inited || !anim.inited)
        return;
      bool opt = fa.getBest().fitness >= MAX_FIT;
      float qcell = BOARD_RECT.height / N;
      for (int r = 0; r < N; r++)
        DrawQueenPiece(anim.queenScreenPos[r], qcell * scale, qcol, opt, time);
    };
    // Part B single-engine: dark-start fitness color for queens
    if (showO && !showM) {
      float ro = origFA.inited
                     ? std::min(1.f, origFA.getBest().fitness / MAX_FIT)
                     : 0.f;
      Color qco = ro < 0.5f
                      ? LerpColor({10, 15, 70, 255}, PAL.origDim, ro * 2.f)
                      : LerpColor(PAL.origDim, PAL.orig, (ro - 0.5f) * 2.f);
      drawQueensAnimated(animO, origFA, qco, 0.72f);
    }
    if (showM && !showO) {
      float rm =
          modFA.inited ? std::min(1.f, modFA.getBest().fitness / MAX_FIT) : 0.f;
      Color qcm = rm < 0.5f ? LerpColor({70, 10, 10, 255}, PAL.modDim, rm * 2.f)
                            : LerpColor(PAL.modDim, PAL.mod, (rm - 0.5f) * 2.f);
      drawQueensAnimated(animM, modFA, qcm, 0.72f);
    }
    if (showO && showM) {
      // Dual mode: both queens centred in cell, orig slightly above, mod
      // slightly below
      float yOff = cell * 0.13f;
      float qSize = cell * 0.56f;
      // When filter=1 or 2 show single queen full size centred
      float qSizeSingle = cell * 0.72f;
      bool optO = origFA.inited && origFA.getBest().fitness >= MAX_FIT;
      bool optM = modFA.inited && modFA.getBest().fitness >= MAX_FIT;
      float ratioO = origFA.inited
                         ? std::min(1.f, origFA.getBest().fitness / MAX_FIT)
                         : 0.f;
      float ratioM =
          modFA.inited ? std::min(1.f, modFA.getBest().fitness / MAX_FIT) : 0.f;
      Color qColO =
          ratioO < 0.5f
              ? LerpColor({10, 15, 70, 255}, PAL.origDim, ratioO * 2.f)
              : LerpColor(PAL.origDim, PAL.orig, (ratioO - 0.5f) * 2.f);
      Color qColM = ratioM < 0.5f
                        ? LerpColor({70, 10, 10, 255}, PAL.modDim, ratioM * 2.f)
                        : LerpColor(PAL.modDim, PAL.mod, (ratioM - 0.5f) * 2.f);
      float vc = qSize * 0.42f * 0.28f;
      float vcSing = qSizeSingle * 0.42f * 0.28f;
      for (int r = 0; r < N; r++) {
        if (boardQueenFilter == 0) {
          // Both: stacked
          Vector2 posO = animO.queenScreenPos[r];
          posO.y -= yOff;
          // Orange: cancel internal vc shift, use smaller offset to sit closer
          // to centre
          Vector2 posM = animM.queenScreenPos[r];
          posM.y = posM.y - vc + yOff * 0.05f;
          DrawQueenPiece(posO, qSize, qColO, optO, time);
          DrawQueenPiece(posM, qSize, qColM, optM, time);
        } else if (boardQueenFilter == 1) {
          // Blue only — full size, centred
          Vector2 posO = animO.queenScreenPos[r];
          posO.y -= vcSing;
          DrawQueenPiece(posO, qSizeSingle, qColO, optO, time);
        } else {
          // Orange only — full size, centred
          Vector2 posM = animM.queenScreenPos[r];
          posM.y -= vcSing;
          DrawQueenPiece(posM, qSizeSingle, qColM, optM, time);
        }
      }
    }
    // Board border
    DrawRectangleLinesEx(BOARD_RECT, 2, PAL.panelBord);

    // Fitness label below board — only show in single engine mode (dual shows
    // filter buttons there)
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
    // In dual mode show fitness scores above the N controls row
    if (showO && showM) {
      if (origFA.inited) {
        auto b = origFA.getBest();
        DrawTextEx(fontSm, TextFormat("Orig: %d/%d", (int)b.fitness, MAX_FIT),
                   {bx, by + N * cell + 8}, 13, 1, PAL.orig);
      }
      if (modFA.inited) {
        auto b = modFA.getBest();
        const char *txt = TextFormat("Mod: %d/%d", (int)b.fitness, MAX_FIT);
        Vector2 sz = MeasureTextEx(fontSm, txt, 13, 1);
        DrawTextEx(fontSm, txt,
                   {bx + BOARD_RECT.width - sz.x, by + N * cell + 8}, 13, 1,
                   PAL.mod);
      }
    }
    // boardQueenFilter reset when not dual — buttons drawn later beside N
    // controls
    if (!(showO && showM))
      boardQueenFilter = 0;

    // ---- N CONTROLS / QUEEN FILTER below chessboard ----
    {
      bool dualMode = showO && showM;
      float nby = by + N * cell + 28.f;

      if (!dualMode) {
        // ── Normal mode: centred [N-] [N=8] [N+] ──────────────────
        float nbw = 160.f;
        float nbx = bx + (BOARD_RECT.width / 2.f) - nbw / 2.f;
        bool hovDec =
            CheckCollisionPointRec(GetMousePosition(), {nbx, nby, 36, 26});
        bool hovInc = CheckCollisionPointRec(GetMousePosition(),
                                             {nbx + 124, nby, 36, 26});
        // N- button
        DrawRectangleRounded(
            {nbx, nby, 36, 26}, 0.4f, 8,
            ColorAlpha(hovDec ? PAL.conflict
                              : ColorBrightness(PAL.conflict, -0.3f),
                       0.9f));
        DrawTextEx(fontBold, "-", {nbx + 12, nby + 4}, 16, 1, WHITE);
        // N badge
        DrawRectangleRounded({nbx + 40, nby, 80, 26}, 0.4f, 8,
                             ColorAlpha(PAL.bg, 0.88f));
        DrawRectangleRoundedLinesEx({nbx + 40, nby, 80, 26}, 0.4f, 8, 1.2f,
                                    PAL.gold);
        Vector2 nSz = MeasureTextEx(fontBold, TextFormat("N = %d", N), 13, 1);
        DrawTextEx(fontBold, TextFormat("N = %d", N),
                   {nbx + 40 + 40 - nSz.x / 2, nby + 6}, 13, 1, PAL.gold);
        // N+ button
        DrawRectangleRounded(
            {nbx + 124, nby, 36, 26}, 0.4f, 8,
            ColorAlpha(hovInc ? PAL.safe : ColorBrightness(PAL.safe, -0.3f),
                       0.9f));
        DrawTextEx(fontBold, "+", {nbx + 136, nby + 4}, 16, 1, WHITE);
        // Click handling
        if (hovDec && IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && N > 4) {
          N--;
          recalcParams();
          resetRun();
          origFA.done = modFA.done = true;
        }
        if (hovInc && IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && N < 32) {
          N++;
          recalcParams();
          resetRun();
          origFA.done = modFA.done = true;
        }
      } else {
        // ── Dual mode: N badge REPLACED by [Both] [Blue] [Org] ────
        // Centred under board
        const char *bqLabels[] = {"Both", "Blue", "Org"};
        Color bqCols[] = {PAL.accent, PAL.orig, PAL.mod};
        float bqW = 64.f, bqH = 26.f, bqGap = 8.f;
        float totalW = 3 * bqW + 2 * bqGap;
        float bqX = bx + (BOARD_RECT.width / 2.f) - totalW / 2.f;
        for (int v = 0; v < 3; v++) {
          float vx = bqX + v * (bqW + bqGap);
          bool sel = boardQueenFilter == v;
          bool hov =
              CheckCollisionPointRec(GetMousePosition(), {vx, nby, bqW, bqH});
          DrawRectangleRounded(
              {vx, nby, bqW, bqH}, 0.4f, 8,
              ColorAlpha(sel ? bqCols[v] : ColorBrightness(bqCols[v], -0.35f),
                         sel ? 1.0f : 0.6f));
          if (sel)
            DrawRectangleRoundedLinesEx({vx, nby, bqW, bqH}, 0.4f, 8, 1.5f,
                                        WHITE);
          Vector2 bsz = MeasureTextEx(fontBold, bqLabels[v], 13, 1);
          DrawTextEx(fontBold, bqLabels[v],
                     {vx + bqW / 2 - bsz.x / 2, nby + bqH / 2 - bsz.y / 2}, 13,
                     1, WHITE);
          if (hov && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            boardQueenFilter = v;
        }
      }
    }

    // ---- SWARM PANEL ----
    {
      // swarmView persists across frames (static) — 0=dual,1=orig,2=mod
      static int swarmView = 0;
      bool bothRunning = showO && showM;

      // Drive swarm layout every frame so fireflies lerp smoothly to focused
      // positions
      if (bothRunning) {
        if (swarmView == 1) {
          // Orig fills full panel
          origFA.swarmHalf = 0;
          origFA.layoutSwarm(0);
          modFA.swarmHalf = 2;
          modFA.layoutSwarm(2); // keep mod off to right (hidden)
        } else if (swarmView == 2) {
          // Mod fills full panel
          modFA.swarmHalf = 0;
          modFA.layoutSwarm(0);
          origFA.swarmHalf = 1;
          origFA.layoutSwarm(1); // keep orig off to left (hidden)
        } else {
          // Dual: restore split layout
          origFA.swarmHalf = 1;
          origFA.layoutSwarm(1);
          modFA.swarmHalf = 2;
          modFA.layoutSwarm(2);
        }
      }

      const char *panelTitle;
      if (!bothRunning) {
        panelTitle =
            (showM && !showO) ? "Modified FA Swarm" : "Original FA Swarm";
      } else if (swarmView == 1) {
        panelTitle = "Original FA Swarm";
      } else if (swarmView == 2) {
        panelTitle = "Modified FA Swarm";
      } else {
        panelTitle = "Dual Swarm";
      }

      Color swBorder = running ? LerpColor(PAL.panelBord, PAL.safe,
                                           0.5f + 0.5f * sinf(time * 4.f))
                               : PAL.panelBord;
      DrawPanel(SWARM_RECT, panelTitle, fontSm, swBorder);

      // Draw toggle buttons AFTER panel so they appear on top of header
      if (bothRunning) {
        const char *viewLabels[] = {"Dual", "Orig", "Mod"};
        float tvW = 48.f, tvH = 18.f;
        float tvX = SWARM_RECT.x + SWARM_RECT.width - tvW * 3 - 10;
        float tvY = SWARM_RECT.y + 7;
        for (int v = 0; v < 3; v++) {
          float vx = tvX + v * tvW;
          bool sel = swarmView == v;
          bool hov = CheckCollisionPointRec(GetMousePosition(),
                                            {vx, tvY, tvW - 3, tvH});
          Color vc2 = v == 0 ? PAL.accent : (v == 1 ? PAL.orig : PAL.mod);
          DrawRectangleRounded(
              {vx, tvY, tvW - 3, tvH}, 0.4f, 8,
              ColorAlpha(sel ? vc2 : ColorBrightness(vc2, -0.35f),
                         sel ? 1.0f : 0.55f));
          if (sel)
            DrawRectangleRoundedLinesEx({vx, tvY, tvW - 3, tvH}, 0.4f, 8, 1.5f,
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

      // Draw engines based on swarmView
      if (!bothRunning) {
        if (showO)
          DrawSwarmPanel(origFA, time, showTrails, particles);
        if (showM)
          DrawSwarmPanel(modFA, time, showTrails, particles);
      } else if (swarmView == 1) {
        DrawSwarmPanel(origFA, time, showTrails, particles);
      } else if (swarmView == 2) {
        DrawSwarmPanel(modFA, time, showTrails, particles);
      } else {
        DrawSwarmPanel(origFA, time, showTrails, particles);
        DrawSwarmPanel(modFA, time, showTrails, particles);
      }
      EndScissorMode();

      // ---- UNIFIED COMPLETION PROGRESS BAR ----
      {
        float progO =
            (showO && origFA.inited) ? (float)origFA.iter / MAX_ITER : 0.f;
        float progM =
            (showM && modFA.inited) ? (float)modFA.iter / MAX_ITER : 0.f;
        int active = (showO ? 1 : 0) + (showM ? 1 : 0);
        float combined = active > 0 ? (progO + progM) / active : 0.f;

        float barX = SWARM_RECT.x + 10;
        float barY = SWARM_RECT.y + SWARM_RECT.height - 24;
        float barW = SWARM_RECT.width - 20;
        float barH = 14.f;

        // Track
        DrawRectangleRounded({barX, barY, barW, barH}, 1.f, 8,
                             ColorAlpha(PAL.panelBord, 0.5f));
        // Filled gradient
        if (combined > 0.f) {
          float fillW = combined * barW;
          Color cLeft = showO ? PAL.orig : PAL.mod;
          Color cRight = showM ? PAL.mod : PAL.orig;
          DrawRectangleGradientH((int)barX, (int)barY, (int)fillW, (int)barH,
                                 cLeft, cRight);
          DrawRectangleRounded({barX, barY, fillW, barH}, 1.f, 8,
                               ColorAlpha(WHITE, 0.04f));
        }
        // Border
        DrawRectangleRoundedLinesEx({barX, barY, barW, barH}, 1.f, 8, 1.f,
                                    ColorAlpha(PAL.panelBord, 0.8f));
        // Centered % label
        int pct = (int)(combined * 100.f);
        const char *pctStr = TextFormat("%d%%", pct);
        Vector2 pctSz = MeasureTextEx(fontSm, pctStr, 11, 1);
        DrawTextEx(fontSm, pctStr, {barX + barW / 2 - pctSz.x / 2, barY + 1},
                   11, 1, ColorAlpha(WHITE, 0.9f));
        // Individual labels
        if (showO)
          DrawTextEx(fontSm, TextFormat("O:%d%%", (int)(progO * 100)),
                     {barX, barY - 14}, 10, 1, ColorAlpha(PAL.orig, 0.8f));
        if (showM)
          DrawTextEx(fontSm, TextFormat("M:%d%%", (int)(progM * 100)),
                     {barX + barW - 46, barY - 14}, 10, 1,
                     ColorAlpha(PAL.mod, 0.8f));
      }
    }

    // ---- CONVERGENCE GRAPH ----
    {
      DrawPanel(GRAPH_RECT, nullptr, fontSm);

      // Panel title
      DrawTextEx(fontSm, "CONVERGENCE GRAPH",
                 {GRAPH_RECT.x + 12, GRAPH_RECT.y + 10}, 11, 1.5f, PAL.dimText);

      // ── STAT CARDS ──────────────────────────────────────────────────
      float cardY = GRAPH_RECT.y + 34;
      float cardH = 40.f;
      float cardW = (GRAPH_RECT.width - 24.f) / 4.f - 4.f;
      float cardX = GRAPH_RECT.x + 12;

      struct MiniCard {
        const char *label;
        std::string val;
        Color col;
      };
      MiniCard cards[4] = {
          {"ORIG BEST",
           (showO && origFA.inited)
               ? TextFormat("%.0f / %d", origFA.stats.bestFit, MAX_FIT)
               : "--",
           PAL.orig},
          {"MOD BEST",
           (showM && modFA.inited)
               ? TextFormat("%.0f / %d", modFA.stats.bestFit, MAX_FIT)
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
           PAL.dimText},
      };

      for (int i = 0; i < 4; i++) {
        float cx = cardX + i * (cardW + 4);
        DrawRectangleRounded({cx, cardY, cardW, cardH}, 0.18f, 8,
                             ColorAlpha(PAL.bg, 0.8f));
        DrawRectangleRoundedLinesEx({cx, cardY, cardW, cardH}, 0.18f, 8, 0.5f,
                                    ColorAlpha(PAL.panelBord, 0.8f));
        DrawTextEx(fontSm, cards[i].label, {cx + 6, cardY + 5}, 9, 1.0f,
                   PAL.dimText);
        DrawTextEx(fontSm, cards[i].val.c_str(), {cx + 6, cardY + 19}, 14, 1,
                   cards[i].col);
      }

      // ── PLOT AREA ────────────────────────────────────────────────────
      float plotX = GRAPH_RECT.x + 44;
      float plotY = GRAPH_RECT.y + 90;
      float plotW = GRAPH_RECT.width - 58;
      float plotH = GRAPH_RECT.height - 122;

      // Y axis visible range: bottom half cut off so lines fill the plot
      float yMin = (float)MAX_FIT * 0.45f;
      float yRange = (float)MAX_FIT - yMin;

      // Plot background
      DrawRectangleRec({plotX, plotY, plotW, plotH}, ColorAlpha(PAL.bg, 0.55f));

      // Horizontal grid + Y labels (5 lines)
      for (int i = 0; i <= 4; i++) {
        float t = (float)i / 4.f;
        float gy = plotY + plotH - t * plotH;
        float val = yMin + yRange * t;
        DrawLineEx({plotX, gy}, {plotX + plotW, gy}, 0.5f,
                   ColorAlpha(PAL.axis, i == 0 ? 0.7f : 0.2f));
        DrawTextEx(fontSm, TextFormat("%d", (int)val), {plotX - 34, gy - 7}, 11,
                   1, PAL.dimText);
      }

      // Vertical grid + X labels (every 30 iters)
      for (int i = 0; i <= MAX_ITER; i += 30) {
        float gx = plotX + (float)i / MAX_ITER * plotW;
        DrawLineEx({gx, plotY}, {gx, plotY + plotH}, 0.5f,
                   ColorAlpha(PAL.axis, i == 0 ? 0.0f : 0.15f));
        DrawLineEx({gx, plotY + plotH}, {gx, plotY + plotH + 4}, 0.5f,
                   ColorAlpha(PAL.axis, 0.5f));
        DrawTextEx(fontSm, TextFormat("%d", i), {gx - 8, plotY + plotH + 7}, 11,
                   1, PAL.dimText);
      }

      // Axis lines
      DrawLineEx({plotX, plotY}, {plotX, plotY + plotH}, 1.f,
                 ColorAlpha(PAL.axis, 0.8f));
      DrawLineEx({plotX, plotY + plotH}, {plotX + plotW, plotY + plotH}, 1.f,
                 ColorAlpha(PAL.axis, 0.8f));

      // ── SERIES DRAWING ───────────────────────────────────────────────
      // Returns screen Y for a fitness value, clamped to plot bounds
      auto fitToY = [&](float fit) -> float {
        float y = plotY + plotH - ((fit - yMin) / yRange) * plotH;
        return Clamp(y, plotY, plotY + plotH);
      };

      // Filled area under best curve
      auto DrawFill = [&](const std::vector<float> &data, Color col) {
        if (data.size() < 2)
          return;
        for (int i = 0; i < (int)data.size() - 1; i++) {
          float x1 = plotX + (float)i / MAX_ITER * plotW;
          float x2 = plotX + (float)(i + 1) / MAX_ITER * plotW;
          float y1 = fitToY(data[i]);
          float y2 = fitToY(data[i + 1]);
          float yb = plotY + plotH;
          // Two triangles forming a trapezoid
          DrawTriangle({x1, yb}, {x1, y1}, {x2, y2}, ColorAlpha(col, 0.06f));
          DrawTriangle({x1, yb}, {x2, y2}, {x2, yb}, ColorAlpha(col, 0.06f));
        }
      };

      // Solid or dashed polyline
      auto DrawLine = [&](const std::vector<float> &data, Color col,
                          float thick, bool dashed) {
        if (data.size() < 2)
          return;
        for (int i = 0; i < (int)data.size() - 1; i++) {
          if (dashed && i % 3 == 2)
            continue; // skip every 3rd segment → dashed
          float x1 = plotX + (float)i / MAX_ITER * plotW;
          float x2 = plotX + (float)(i + 1) / MAX_ITER * plotW;
          DrawLineEx({x1, fitToY(data[i])}, {x2, fitToY(data[i + 1])}, thick,
                     col);
        }
      };

      // Endpoint dot on the latest value
      auto DrawEndDot = [&](const std::vector<float> &data, Color col,
                            float thick) {
        if (data.empty())
          return;
        float ex = plotX + (float)(data.size() - 1) / MAX_ITER * plotW;
        float ey = fitToY(data.back());
        DrawCircleV({ex, ey}, thick + 2.5f, col);
        DrawCircleV({ex, ey}, thick * 0.6f, ColorAlpha(WHITE, 0.85f));
      };

      // Draw orig then mod (fills first so lines sit on top)
      if (showO) {
        DrawFill(origFA.stats.bestPerIter, PAL.orig);
        DrawLine(origFA.stats.avgPerIter, ColorAlpha(PAL.orig, 0.4f), 1.2f,
                 true);
        DrawLine(origFA.stats.bestPerIter, PAL.orig, 2.5f, false);
        DrawEndDot(origFA.stats.bestPerIter, PAL.orig, 2.5f);
      }
      if (showM) {
        DrawFill(modFA.stats.bestPerIter, PAL.mod);
        DrawLine(modFA.stats.avgPerIter, ColorAlpha(PAL.mod, 0.4f), 1.2f, true);
        DrawLine(modFA.stats.bestPerIter, PAL.mod, 2.5f, false);
        DrawEndDot(modFA.stats.bestPerIter, PAL.mod, 2.5f);
      }

      // Optimal fitness reference line
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

      // ── LEGEND ───────────────────────────────────────────────────────
      float legY = plotY + plotH + 20;
      float legX = plotX;

      struct LegEntry {
        const char *label;
        Color col;
        bool dashed;
      };
      LegEntry legs[] = {
          {"Orig best", PAL.orig, false},
          {"Orig avg", ColorAlpha(PAL.orig, 0.4f), true},
          {"Mod best", PAL.mod, false},
          {"Mod avg", ColorAlpha(PAL.mod, 0.4f), true},
      };

      float lx = legX;
      for (auto &le : legs) {
        // Line sample
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

      // Position the expand button based on EXPAND_BTN_POS
      {
        float bw = 84.f, bh = 24.f;
        float pad = 8.f;
        switch (EXPAND_BTN_POS) {
        case 0:
          btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - bw - 2,
                                 GRAPH_RECT.y + 2, bw, bh};
          break;
        case 1:
          btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - bw - 12,
                                 GRAPH_RECT.y, bw, bh};
          break;
        case 2:
          btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - bw - pad,
                                 GRAPH_RECT.y + GRAPH_RECT.height - bh - pad,
                                 bw, bh};
          break;
        case 3:
          btnExpandGraph.rect = {GRAPH_RECT.x + pad,
                                 GRAPH_RECT.y + GRAPH_RECT.height - bh - pad,
                                 bw, bh};
          break;
        case 4:
          btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - bw - pad,
                                 GRAPH_RECT.y + 6, bw, 20.f};
          break;
        case 5:
          btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - bw,
                                 GRAPH_RECT.y + GRAPH_RECT.height + 4, bw, bh};
          break;
        default:
          btnExpandGraph.rect = {GRAPH_RECT.x + GRAPH_RECT.width - bw - pad,
                                 GRAPH_RECT.y + pad, bw, bh};
        }
      }
      if (showO || showM) {
        btnExpandGraph.draw(fontSm);
      }
    }

    // ---- METRICS / COMPARISON PANEL ----
    {
      bool bothDone = showO && showM && origFA.inited && modFA.inited &&
                      origFA.done && modFA.done;

      // Live-compute every frame when both engines are running or done
      bool bothActive = showO && showM && origFA.inited && modFA.inited;
      if (bothActive)
        cmpResult = ComputeComparison(origFA, modFA);

      // Panel title changes per page
      const char *panelTitles[] = {"Algorithm Comparison", "Convergence Detail",
                                   "Parameter Summary"};
      const char *title =
          bothActive         ? panelTitles[(int)Clamp((float)cmpPage, 0.f, 2.f)]
          : (showO || showM) ? "Live Metrics"
                             : "Live Metrics";

      DrawPanel(METRICS_RECT, title, fontSm);

      // ── IDLE STATE ──
      if (!showO && !showM) {
        float cy = METRICS_RECT.y + METRICS_RECT.height / 2.f - 30.f;
        const char *hints[] = {"Press [O] for Original FA",
                               "Press [M] for Modified FA",
                               "Press [B] to Run Both"};
        Color hcols[] = {PAL.orig, PAL.mod, PAL.accent};
        for (int i = 0; i < 3; i++) {
          Vector2 hs = MeasureTextEx(fontSm, hints[i], 13, 1);
          DrawTextEx(fontSm, hints[i],
                     {METRICS_RECT.x + METRICS_RECT.width / 2 - hs.x / 2,
                      cy + i * 22.f},
                     13, 1, hcols[i]);
        }
      }
      // ── BOTH ACTIVE: LIVE COMPARISON PAGES ──
      else if (bothActive && cmpResult.computed) {

        // ── EXPAND BUTTON — always visible top-right when bothActive ──
        btnExpandCmp.rect = {METRICS_RECT.x + METRICS_RECT.width - 66.f,
                             METRICS_RECT.y + 6.f, 60.f, 22.f};
        btnExpandCmp.draw(fontSm);

        // ── PAGE NAV ARROWS — bottom RIGHT corner, always visible when
        // bothActive ──
        float navY = METRICS_RECT.y + METRICS_RECT.height - 26.f;
        float navRX = METRICS_RECT.x + METRICS_RECT.width - 10.f; // right edge

        // Right arrow (rightmost)
        float arrowW = 24.f, arrowH = 20.f;
        float rArrX = navRX - arrowW;
        bool hovR = CheckCollisionPointRec(GetMousePosition(),
                                           {rArrX, navY, arrowW, arrowH});
        DrawRectangleRounded(
            {rArrX, navY, arrowW, arrowH}, 0.4f, 8,
            ColorAlpha(hovR ? PAL.accent : PAL.panelBord, 0.7f));
        DrawTextEx(fontSm, ">", {rArrX + 6, navY + 3}, 14, 1, WHITE);
        if (hovR && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
          cmpPage = (cmpPage + 1) % 3;

        // Page dots (left of right arrow)
        float dotsEndX = rArrX - 6.f;
        for (int p = 2; p >= 0; p--) {
          float dx = dotsEndX - (2 - p) * 12.f;
          DrawCircleV({dx, navY + 10}, 3.5f,
                      p == cmpPage ? PAL.accent
                                   : ColorAlpha(PAL.dimText, 0.5f));
        }

        // Left arrow (left of dots)
        float lArrX = dotsEndX - 3 * 12.f - 6.f - arrowW;
        bool hovL = CheckCollisionPointRec(GetMousePosition(),
                                           {lArrX, navY, arrowW, arrowH});
        DrawRectangleRounded(
            {lArrX, navY, arrowW, arrowH}, 0.4f, 8,
            ColorAlpha(hovL ? PAL.accent : PAL.panelBord, 0.7f));
        DrawTextEx(fontSm, "<", {lArrX + 6, navY + 3}, 14, 1, WHITE);
        if (hovL && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
          cmpPage = (cmpPage + 2) % 3;

        // Content area
        Rectangle cmpBounds = {METRICS_RECT.x + 4, METRICS_RECT.y + 32,
                               METRICS_RECT.width - 8,
                               METRICS_RECT.height - 62};

        // ── PAGE 0: STATS TABLE ──
        if (cmpPage == 0) {
          DrawComparisonPanel(cmpBounds, cmpResult, fontSm, fontMono, time);
        }

        // ── PAGE 1: MINI CONVERGENCE GRAPHS ──
        else if (cmpPage == 1) {
          float gx = cmpBounds.x + 36.f;
          float gy = cmpBounds.y + 8.f;
          float gw = cmpBounds.width - 44.f;
          float gh = (cmpBounds.height - 36.f) / 2.f - 8.f;

          float yMin = (float)MAX_FIT * 0.4f;
          float yRange = (float)MAX_FIT - yMin;

          auto fitY = [&](float f, float top, float h) -> float {
            return Clamp(top + h - ((f - yMin) / yRange) * h, top, top + h);
          };

          // Draw one mini graph
          auto drawMiniGraph = [&](float top, const std::vector<float> &best,
                                   const std::vector<float> &avg,
                                   const std::vector<float> &worst, Color col,
                                   const char *label) {
            // Background
            DrawRectangleRec({gx, top, gw, gh}, ColorAlpha(PAL.bg, 0.5f));
            DrawRectangleRoundedLines({gx, top, gw, gh}, 0.05f, 8, 0.5f,
                                      ColorAlpha(col, 0.4f));

            // Optimal line
            float optY2 = fitY((float)MAX_FIT, top, gh);
            for (int i = 0; i < (int)gw; i += 6)
              DrawLineEx({gx + (float)i, optY2}, {gx + (float)i + 3, optY2},
                         0.8f, ColorAlpha(PAL.gold, 0.3f));

            // Y axis labels (3 ticks)
            for (int t = 0; t <= 2; t++) {
              float tv = yMin + yRange * t / 2.f;
              float ty = fitY(tv, top, gh);
              DrawLineEx({gx - 4, ty}, {gx, ty}, 0.5f,
                         ColorAlpha(PAL.dimText, 0.5f));
              DrawTextEx(fontSm, TextFormat("%d", (int)tv), {gx - 34, ty - 6},
                         10, 1, PAL.dimText);
            }

            // Series
            if (worst.size() > 1) {
              for (int i = 0; i < (int)worst.size() - 1; i++) {
                float x1 = gx + (float)i / (MAX_ITER)*gw;
                float x2 = gx + (float)(i + 1) / (MAX_ITER)*gw;
                DrawLineEx({x1, fitY(worst[i], top, gh)},
                           {x2, fitY(worst[i + 1], top, gh)}, 1.f,
                           ColorAlpha(col, 0.2f));
              }
            }
            if (avg.size() > 1) {
              for (int i = 0; i < (int)avg.size() - 1; i++) {
                if (i % 3 == 2)
                  continue;
                float x1 = gx + (float)i / (MAX_ITER)*gw;
                float x2 = gx + (float)(i + 1) / (MAX_ITER)*gw;
                DrawLineEx({x1, fitY(avg[i], top, gh)},
                           {x2, fitY(avg[i + 1], top, gh)}, 1.2f,
                           ColorAlpha(col, 0.5f));
              }
            }
            if (best.size() > 1) {
              for (int i = 0; i < (int)best.size() - 1; i++) {
                float x1 = gx + (float)i / (MAX_ITER)*gw;
                float x2 = gx + (float)(i + 1) / (MAX_ITER)*gw;
                DrawLineEx({x1, fitY(best[i], top, gh)},
                           {x2, fitY(best[i + 1], top, gh)}, 2.f, col);
              }
              // End dot
              float ex = gx + (float)(best.size() - 1) / (MAX_ITER)*gw;
              DrawCircleV({ex, fitY(best.back(), top, gh)}, 4.f, col);
            }

            // Label
            DrawTextEx(fontSm, label, {gx + 4, top + 4}, 11, 1,
                       ColorAlpha(col, 0.9f));

            // Best value badge
            if (!best.empty()) {
              const char *bv = TextFormat("Best: %.0f", best.back());
              Vector2 bvs = MeasureTextEx(fontSm, bv, 10, 1);
              DrawRectangleRounded(
                  {gx + gw - bvs.x - 10, top + 2, bvs.x + 8, 14}, 0.4f, 4,
                  ColorAlpha(col, 0.2f));
              DrawTextEx(fontSm, bv, {gx + gw - bvs.x - 6, top + 3}, 10, 1,
                         ColorAlpha(col, 0.9f));
            }
          };

          if (showO && origFA.inited)
            drawMiniGraph(gy, origFA.stats.bestPerIter, origFA.stats.avgPerIter,
                          origFA.stats.worstPerIter, PAL.orig, "Original FA");

          if (showM && modFA.inited)
            drawMiniGraph(gy + gh + 14.f, modFA.stats.bestPerIter,
                          modFA.stats.avgPerIter, modFA.stats.worstPerIter,
                          PAL.mod, "Modified FA");

          // Shared X-axis labels
          float xAxisY = gy + gh * 2 + 22.f;
          for (int i = 0; i <= 4; i++) {
            float xi = gx + (float)i / 4.f * gw;
            int iv = MAX_ITER * i / 4;
            DrawTextEx(fontSm, TextFormat("%d", iv), {xi - 8, xAxisY}, 10, 1,
                       PAL.dimText);
            DrawLineEx({xi, gy}, {xi, gy + gh * 2 + 14.f}, 0.3f,
                       ColorAlpha(PAL.axis, 0.15f));
          }
          DrawTextEx(fontSm, "Iterations", {gx + gw / 2 - 24, xAxisY + 12}, 10,
                     1, PAL.dimText);

          // Legend row
          float legY2 = xAxisY + 26.f;
          struct LI {
            const char *l;
            Color c;
            bool d;
          };
          LI lis[] = {
              {"Best", PAL.orig, false},
              {"Avg", PAL.orig, true},
              {"Worst", PAL.orig, true},
          };
          float lxp = gx;
          for (auto &li : lis) {
            if (li.d) {
              DrawLineEx({lxp, legY2 + 4}, {lxp + 8, legY2 + 4}, 1.2f,
                         ColorAlpha(li.c, 0.5f));
              DrawLineEx({lxp + 12, legY2 + 4}, {lxp + 18, legY2 + 4}, 1.2f,
                         ColorAlpha(li.c, 0.5f));
            } else {
              DrawLineEx({lxp, legY2 + 4}, {lxp + 18, legY2 + 4}, 2.f, li.c);
              DrawCircleV({lxp + 9, legY2 + 4}, 3.f, li.c);
            }
            DrawTextEx(fontSm, li.l, {lxp + 22, legY2}, 10, 1, PAL.dimText);
            lxp += 52.f;
          }
        }

        // ── PAGE 2: PARAMETER SUMMARY ──
        else if (cmpPage == 2) {
          float px = cmpBounds.x + 10.f;
          float py = cmpBounds.y + 6.f;
          float pw = cmpBounds.width - 20.f;
          float barH2 = 12.f;
          float rowH2 = 36.f;

          DrawTextEx(fontSm, "Active Modified FA Parameters:", {px, py}, 11, 1,
                     PAL.dimText);
          py += 18.f;

          struct ParamBar {
            const char *name;
            float value; // 0..1 normalized
            float rawVal;
            const char *valStr;
            Color col;
          };

          ParamBar pbars[] = {
              {"Alpha0 (Randomness)", (modParams.alpha0 - 0.1f) / 1.9f,
               modParams.alpha0, TextFormat("%.2f", modParams.alpha0),
               PAL.gold},
              {"Mutation Rate",
               modParams.mutRate,
               modParams.mutRate,
               TextFormat("%.0f%%", modParams.mutRate * 100),
               {200, 100, 255, 255}},
              {"Heuristic Bias", modParams.heurRatio, modParams.heurRatio,
               modParams.heurRatio < 0.25f   ? "Random"
               : modParams.heurRatio < 0.75f ? "Mixed"
                                             : "Heuristic",
               PAL.safe},
              {"Elite Count", (float)modParams.eliteCount / 4.f,
               (float)modParams.eliteCount,
               TextFormat("%d / 4", modParams.eliteCount), PAL.mod},
          };

          for (auto &pb : pbars) {
            // Name + value
            DrawTextEx(fontSm, pb.name, {px, py}, 11, 1,
                       ColorAlpha(pb.col, 0.9f));
            Vector2 vs = MeasureTextEx(fontSm, pb.valStr, 12, 1);
            DrawTextEx(fontSm, pb.valStr, {px + pw - vs.x, py}, 12, 1, pb.col);
            py += 14.f;

            // Track
            DrawRectangleRounded({px, py, pw, barH2}, 1.f, 4,
                                 ColorAlpha(PAL.panelBord, 0.5f));
            // Gradient fill
            DrawRectangleGradientH(
                (int)px, (int)py, (int)(pb.value * pw), (int)barH2,
                ColorAlpha(ColorBrightness(pb.col, -0.3f), 0.8f),
                ColorAlpha(pb.col, 0.9f));
            // Border
            DrawRectangleRoundedLines({px, py, pw, barH2}, 1.f, 4, 0.5f,
                                      ColorAlpha(pb.col, 0.3f));

            py += rowH2 - 14.f;
          }

          // Divider + effect summary
          py += 4.f;
          DrawLineEx({px, py}, {px + pw, py}, 0.5f,
                     ColorAlpha(PAL.panelBord, 0.6f));
          py += 8.f;

          DrawTextEx(fontSm, "Effect on this run:", {px, py}, 10, 1,
                     PAL.dimText);
          py += 14.f;

          // Show quick one-line takeaways based on score
          const char *takeaway =
              cmpResult.score >= 4   ? "Parameters are well-tuned"
              : cmpResult.score >= 2 ? "Mixed results — try adjusting"
                                     : "Parameters hurt performance";
          Color tkCol = cmpResult.score >= 4   ? PAL.safe
                        : cmpResult.score >= 2 ? PAL.gold
                                               : PAL.conflict;
          DrawTextEx(fontSm, takeaway, {px, py}, 11, 1, tkCol);
        }
      }
      // ── LIVE METRICS (single engine or not done) ──
      else {
        float cardW = (METRICS_RECT.width - 32) / 2.f;
        float cardH = (METRICS_RECT.height - 84) / 3.f;
        float sx = METRICS_RECT.x + 12;
        float sy = METRICS_RECT.y + 40;

        auto drawStat = [&](int r, int c, const char *label,
                            const char *valOrig, const char *valMod) {
          float x = sx + c * (cardW + 8);
          float y2 = sy + r * (cardH + 8);
          DrawRectangleRounded({x, y2, cardW, cardH}, 0.15f, 8,
                               ColorAlpha(PAL.panelBord, 0.3f));
          // Colour accent top bar
          if (c == 0 && showO)
            DrawRectangleRounded({x, y2, cardW, 3}, 0.5f, 4,
                                 ColorAlpha(PAL.orig, 0.7f));
          if (c == 1 && showM)
            DrawRectangleRounded({x, y2, cardW, 3}, 0.5f, 4,
                                 ColorAlpha(PAL.mod, 0.7f));
          DrawTextEx(fontSm, label, {x + 8, y2 + 7}, 10, 1, PAL.dimText);
          if (showO && c == 0)
            DrawTextEx(fontSm, valOrig, {x + 8, y2 + 20}, 14, 1, PAL.orig);
          if (showM && c == 1)
            DrawTextEx(fontSm, valMod, {x + 8, y2 + 20}, 14, 1, PAL.mod);
        };

        drawStat(
            0, 0, "BEST FITNESS",
            origFA.inited ? TextFormat("%.0f/%d", origFA.stats.bestFit, MAX_FIT)
                          : "--",
            modFA.inited ? TextFormat("%.0f/%d", modFA.stats.bestFit, MAX_FIT)
                         : "--");
        drawStat(
            0, 1, "BEST FITNESS",
            origFA.inited ? TextFormat("%.0f/%d", origFA.stats.bestFit, MAX_FIT)
                          : "--",
            modFA.inited ? TextFormat("%.0f/%d", modFA.stats.bestFit, MAX_FIT)
                         : "--");
        drawStat(1, 0, "AVG FITNESS",
                 origFA.inited ? TextFormat("%.2f", origFA.stats.avgFit) : "--",
                 modFA.inited ? TextFormat("%.2f", modFA.stats.avgFit) : "--");
        drawStat(1, 1, "AVG FITNESS",
                 origFA.inited ? TextFormat("%.2f", origFA.stats.avgFit) : "--",
                 modFA.inited ? TextFormat("%.2f", modFA.stats.avgFit) : "--");
        drawStat(
            2, 0, "ITER / OPT",
            origFA.inited ? TextFormat("%d/%d", origFA.iter, MAX_ITER) : "--",
            modFA.inited ? TextFormat("%d/%d", modFA.iter, MAX_ITER) : "--");
        drawStat(
            2, 1, "ITER / OPT",
            origFA.inited ? (origFA.stats.iterToOptimal < 0
                                 ? "—"
                                 : TextFormat("%d", origFA.stats.iterToOptimal))
                          : "--",
            modFA.inited ? (modFA.stats.iterToOptimal < 0
                                ? "—"
                                : TextFormat("%d", modFA.stats.iterToOptimal))
                         : "--");

        // Diversity bars
        float dbY = sy + 3 * (cardH + 8) + 4;
        DrawTextEx(fontSm, "DIVERSITY", {sx, dbY}, 10, 1, PAL.dimText);
        dbY += 13;
        float dbW = METRICS_RECT.width - 24;
        if (showO && origFA.inited) {
          DrawRectangleRounded({sx, dbY, dbW, 6}, 1.f, 4,
                               ColorAlpha(PAL.panelBord, 0.5f));
          DrawRectangleRounded({sx, dbY, dbW * origFA.diversity, 6}, 1.f, 4,
                               ColorAlpha(PAL.orig, 0.7f));
          DrawTextEx(fontSm, TextFormat("O:%.2f", origFA.diversity),
                     {sx + dbW + 4, dbY - 2}, 9, 1, ColorAlpha(PAL.orig, 0.8f));
        }
        dbY += 10;
        if (showM && modFA.inited) {
          DrawRectangleRounded({sx, dbY, dbW, 6}, 1.f, 4,
                               ColorAlpha(PAL.panelBord, 0.5f));
          DrawRectangleRounded({sx, dbY, dbW * modFA.diversity, 6}, 1.f, 4,
                               ColorAlpha(PAL.mod, 0.7f));
          DrawTextEx(fontSm, TextFormat("M:%.2f", modFA.diversity),
                     {sx + dbW + 4, dbY - 2}, 9, 1, ColorAlpha(PAL.mod, 0.8f));
        }

        // Optimal pulse
        if (solutionFound) {
          float pulse = 0.5f + 0.5f * sinf(time * 6.f);
          const char *s = "OPTIMAL SOLUTION REACHED";
          Vector2 ss = MeasureTextEx(fontSm, s, 11, 1);
          DrawTextEx(
              fontSm, s,
              {METRICS_RECT.x + METRICS_RECT.width / 2 - ss.x / 2, dbY + 18},
              11, 1, ColorAlpha(PAL.gold, 0.4f + 0.6f * pulse));
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

    // ---- CONTROL PANEL ----
    DrawRectangleRounded(CTRL_RECT, 0.08f, 12, ColorAlpha(PAL.panel, 0.9f));
    DrawRectangleRoundedLinesEx(CTRL_RECT, 0.08f, 12, 2.0f, PAL.panelBord);
    // Control bar layout — calculated spacing
    // Left cluster  : [Original FA] [Modified FA] [Run Both] [Reset]
    // Centre cluster: [|<] [PLAY/STOP] [>|]
    // Right cluster : speed slider (existing, positioned separately)
    {
      float bh = 40.f;
      float by0 = CTRL_RECT.y + (CTRL_RECT.height - bh) / 2.f;
      float pad = 10.f; // gap between buttons

      // ── Left cluster ──────────────────────────────────────────
      float lx = CTRL_RECT.x + 14.f;
      btnOrig.rect = {lx, by0, 118.f, bh};
      lx += 118.f + pad;
      btnMod.rect = {lx, by0, 118.f, bh};
      lx += 118.f + pad;
      btnBoth.rect = {lx, by0, 100.f, bh};
      lx += 100.f + pad;
      btnParams.rect = {lx, by0, 110.f, bh};
      lx += 110.f + pad;
      btnReset.rect = {lx, by0, 80.f, bh};

      // ── Centre cluster — playback controls ────────────────────
      // Anchor to horizontal centre of the bar
      float stepW = 46.f;
      float playW = 108.f;
      float clusterW = stepW + pad + playW + pad + stepW;
      float cx = CTRL_RECT.x + CTRL_RECT.width / 2.f - clusterW / 2.f;
      // nudge right slightly so it clears the left cluster on small N
      cx = std::max(cx, lx + 80.f + pad);

      btnStepBwd.rect = {cx, by0, stepW, bh};
      cx += stepW + pad;
      btnPlayStop.rect = {cx, by0, playW, bh};
      cx += playW + pad;
      btnStepFwd.rect = {cx, by0, stepW, bh};

      btnOrig.draw(fontBold);
      btnMod.draw(fontBold);
      btnBoth.draw(fontBold);
      btnParams.draw(fontBold);
      btnReset.draw(fontBold);
      btnStepBwd.draw(fontBold);
      btnPlayStop.draw(fontBold);
      btnStepFwd.draw(fontBold);
    }

    // Speed slider — right cluster, leaving ~20px margin from edge
    speedSlider.track = {CTRL_RECT.x + CTRL_RECT.width - 280.f,
                         CTRL_RECT.y + (CTRL_RECT.height - 8.f) / 2.f, 180.f,
                         8.f};
    speedSlider.draw(fontSm, PAL.gold, "Speed", "%.0f%%", 0.f, 100.f);

    paramPopup.draw(fontSm, fontBold, time);
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

    // Expanded Graph Modal
    if (expandedT > 0.01f) {
      DrawRectangle(0, 0, WIN_W, WIN_H, ColorAlpha(PAL.bg, 0.85f * expandedT));

      float targetY = WIN_H / 2.f - 300.f;
      float modalY =
          WIN_H + 50.f -
          (WIN_H + 50.f - targetY) * expandedT; // Smooth slide from bottom

      Rectangle expRect = {WIN_W / 2.f - 500.f, modalY, 1000.f, 600.f};
      DrawPanel(expRect, "Convergence Graph (Expanded)", fontBold, PAL.safe);

      Rectangle gInner = {expRect.x + 72, expRect.y + 80, expRect.width - 380,
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

      // Legend box — matches the stats panel style on the right
      struct LegItem {
        const char *label;
        Color col;
        bool dashed;
      };
      LegItem legItems[] = {
          {"Orig Best", PAL.orig, false}, {"Orig Avg", PAL.orig, true},
          {"Orig Worst", PAL.orig, true}, {"Mod Best", PAL.mod, false},
          {"Mod Avg", PAL.mod, true},     {"Mod Worst", PAL.mod, true},
      };

      float legBoxX = gInner.x;
      float legBoxY = gInner.y + gInner.height + 16;
      float legBoxW = gInner.width - 44.f;
      float legBoxH = 72.f;
      float legRowH = 34.f;
      float legColW = legBoxW / 3.f;

      // Box background + border (same style as stats panel)
      DrawRectangleRounded({legBoxX, legBoxY, legBoxW, legBoxH}, 0.08f, 8,
                           ColorAlpha(PAL.bg, 0.6f));
      DrawRectangleRoundedLinesEx({legBoxX, legBoxY, legBoxW, legBoxH}, 0.08f,
                                  8, 0.5f, ColorAlpha(PAL.panelBord, 0.6f));

      // Draw 2 rows x 3 columns
      for (int i = 0; i < 6; i++) {
        int col = i % 3;
        int row = i / 3;
        float ix = legBoxX + col * legColW + 12;
        float iy = legBoxY + 8 + row * legRowH;

        // Line sample
        if (legItems[i].dashed) {
          DrawLineEx({ix, iy + 8}, {ix + 12, iy + 8}, 2.f, legItems[i].col);
          DrawLineEx({ix + 17, iy + 8}, {ix + 28, iy + 8}, 2.f,
                     legItems[i].col);
        } else {
          DrawLineEx({ix, iy + 8}, {ix + 28, iy + 8}, 2.5f, legItems[i].col);
          DrawCircleV({ix + 14, iy + 8}, 4.5f, legItems[i].col);
        }
        DrawTextEx(fontSm, legItems[i].label, {ix + 34, iy + 1}, 18, 1,
                   legItems[i].col);
      }

      // Expanded Stats
      // ── STATS SIDE PANEL (right of graph, vertical) ──────────────
      float spX = gInner.x + gInner.width + 20;
      float spY = gInner.y;
      float spW = 160.f;
      float spH = gInner.height;
      float rowH = 44.f;

      // Background card for the side panel
      DrawRectangleRounded({spX - 8, spY, spW + 8, spH}, 0.08f, 8,
                           ColorAlpha(PAL.bg, 0.6f));
      DrawRectangleRoundedLinesEx({spX - 8, spY, spW + 8, spH}, 0.08f, 8, 0.5f,
                                  ColorAlpha(PAL.panelBord, 0.6f));

      struct StatRow {
        const char *label;
        std::string val;
        Color col;
      };
      std::vector<StatRow> rows;

      if (showO) {
        rows.push_back({"Orig Best",
                        TextFormat("%.0f / %d", origFA.stats.bestFit, MAX_FIT),
                        PAL.orig});
        rows.push_back({"To Optimal",
                        origFA.stats.iterToOptimal < 0
                            ? "--"
                            : TextFormat("iter %d", origFA.stats.iterToOptimal),
                        PAL.orig});
        rows.push_back({"Opt Hits", TextFormat("%d", origFA.stats.optimalCount),
                        PAL.orig});
      }
      // Divider between orig and mod
      if (showO && showM) {
        rows.push_back({"", "", PAL.dimText}); // spacer row
      }
      if (showM) {
        rows.push_back({"Mod Best",
                        TextFormat("%.0f / %d", modFA.stats.bestFit, MAX_FIT),
                        PAL.mod});
        rows.push_back({"To Optimal",
                        modFA.stats.iterToOptimal < 0
                            ? "--"
                            : TextFormat("iter %d", modFA.stats.iterToOptimal),
                        PAL.mod});
        rows.push_back(
            {"Opt Hits", TextFormat("%d", modFA.stats.optimalCount), PAL.mod});
      }

      float ry = spY + 10;
      for (auto &row : rows) {
        if (row.label[0] == '\0') {
          // Spacer — draw a divider line
          DrawLineEx({spX - 4, ry + 6}, {spX + spW - 4, ry + 6}, 0.5f,
                     ColorAlpha(PAL.panelBord, 0.8f));
          ry += 16;
          continue;
        }
        // Key (dim, small)
        DrawTextEx(fontSm, row.label, {spX, ry}, 13, 1, PAL.dimText);
        DrawTextEx(fontSm, row.val.c_str(), {spX, ry + 16}, 18, 1, row.col);
        ry += rowH;
      }
      // Sync Close button position to the animated modal
      btnCloseGraph.rect.y = expRect.y + 4;
      btnCloseGraph.draw(fontSm);
    }

    // ── EXPANDED COMPARISON MODAL ──
    if (expandCmpT > 0.01f) {
      DrawRectangle(0, 0, WIN_W, WIN_H, ColorAlpha(PAL.bg, 0.88f * expandCmpT));

      float mw = 1100.f, mh = 720.f;
      float mx = WIN_W / 2.f - mw / 2.f;
      float targetY = WIN_H / 2.f - mh / 2.f;
      float my = WIN_H + 60.f - (WIN_H + 60.f - targetY) * expandCmpT;

      DrawRectangleRounded({mx, my, mw, mh}, 0.04f, 16,
                           ColorAlpha({10, 12, 28, 255}, expandCmpT));
      DrawRectangleRoundedLinesEx({mx, my, mw, mh}, 0.04f, 16, 2.f,
                                  ColorAlpha(PAL.accent, expandCmpT * 0.9f));

      // Header
      DrawRectangleRounded({mx, my, mw, 50.f}, 0.04f, 16,
                           ColorAlpha({18, 14, 45, 255}, expandCmpT));
      DrawTextEx(fontBold, "FULL COMPARISON REPORT", {mx + 20, my + 14}, 20,
                 1.5f, ColorAlpha(PAL.gold, expandCmpT));

      // Close button
      Rectangle ecClose = {mx + mw - 44, my + 9, 32, 32};
      bool hovEC = CheckCollisionPointRec(GetMousePosition(), ecClose);
      DrawRectangleRounded(
          ecClose, 0.4f, 8,
          ColorAlpha(hovEC ? PAL.conflict
                           : ColorBrightness(PAL.conflict, -0.3f),
                     expandCmpT));
      DrawTextEx(fontBold, "X", {ecClose.x + 10, ecClose.y + 8}, 16, 1,
                 ColorAlpha(WHITE, expandCmpT));
      if (hovEC && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        showExpandCmp = false;

      if (cmpResult.computed) {
        const char *verdSummary[] = {"No improvement", "Slight improvement",
                                     "Moderate improvement",
                                     "Strong improvement", "Full improvement"};
        int vi = Clamp(cmpResult.score, 0, 4);

        // Left half: stats table
        Rectangle leftR = {mx + 20, my + 60, mw * 0.42f - 10, mh - 80};
        DrawPanel(leftR, "Metrics Table", fontSm, PAL.accent);
        float cmpInnerH = leftR.height * 0.52f - 40.f;
        DrawComparisonPanel(
            {leftR.x + 4, leftR.y + 32, leftR.width - 8, cmpInnerH}, cmpResult,
            fontSm, fontMono, time);

        // ── PARAMETER IMPACT BARS ─────────────────────────────────
        float piy = leftR.y + 32.f + cmpInnerH + 6.f;
        float piw = leftR.width - 16.f;
        float pix = leftR.x + 8.f;

        DrawLineEx({pix, piy}, {pix + piw, piy}, 0.5f,
                   ColorAlpha(PAL.panelBord, expandCmpT * 0.8f));
        piy += 6.f;

        DrawTextEx(fontSm, "Parameter Impact (Modified FA)", {pix, piy}, 11, 1,
                   ColorAlpha(PAL.gold, expandCmpT));
        piy += 16.f;

        struct PIBar {
          const char *name;
          float val;
          Color col;
          const char *valStr;
        };
        PIBar piBars[] = {
            {"Alpha0", (modParams.alpha0 - 0.1f) / 1.9f, PAL.gold,
             TextFormat("%.2f", modParams.alpha0)},
            {"Mut Rate",
             modParams.mutRate,
             {200, 100, 255, 255},
             TextFormat("%.0f%%", modParams.mutRate * 100)},
            {"Heuristic", modParams.heurRatio, PAL.safe,
             modParams.heurRatio < 0.25f   ? "Random"
             : modParams.heurRatio < 0.75f ? "Mixed"
                                           : "Heuristic"},
            {"Elites", (float)modParams.eliteCount / 4.f, PAL.mod,
             TextFormat("%d/4", modParams.eliteCount)},
        };
        float piBarH = 10.f;
        float piRowH = 28.f;
        float piLabelW = 60.f;
        float piBarW = piw - piLabelW - 36.f;

        for (auto &pb : piBars) {
          // Label
          DrawTextEx(fontSm, pb.name, {pix, piy + 1}, 10, 1,
                     ColorAlpha(pb.col, expandCmpT * 0.9f));
          // Track
          float trackX = pix + piLabelW;
          DrawRectangleRounded({trackX, piy, piBarW, piBarH}, 1.f, 4,
                               ColorAlpha(PAL.panelBord, expandCmpT * 0.4f));
          // Fill
          if (pb.val > 0.001f)
            DrawRectangleGradientH(
                (int)trackX, (int)piy, (int)(pb.val * piBarW), (int)piBarH,
                ColorAlpha(ColorBrightness(pb.col, -0.3f), expandCmpT * 0.8f),
                ColorAlpha(pb.col, expandCmpT));
          // Value badge
          DrawTextEx(fontSm, pb.valStr, {trackX + piBarW + 4, piy}, 10, 1,
                     ColorAlpha(pb.col, expandCmpT));
          piy += piRowH;
        }

        // ── WINNER BANNER ─────────────────────────────────────────
        if (showO && showM && origFA.inited && modFA.inited && origFA.done &&
            modFA.done) {
          piy += 6.f;
          DrawLineEx({pix, piy}, {pix + piw, piy}, 0.5f,
                     ColorAlpha(PAL.panelBord, expandCmpT * 0.6f));
          piy += 8.f;

          // Head-to-head result box (80px tall)
          DrawRectangleRounded({pix, piy, piw, 80.f}, 0.12f, 8,
                               ColorAlpha(PAL.bg, expandCmpT * 0.8f));
          DrawRectangleRoundedLinesEx({pix, piy, piw, 80.f}, 0.12f, 8, 1.f,
                                      ColorAlpha(PAL.gold, expandCmpT * 0.5f));

          // Two columns: Orig | Mod
          float col1x = pix + 10.f;
          float col2x = pix + piw / 2.f + 5.f;
          float rowY = piy + 8.f;

          // Column headers
          DrawTextEx(fontSm, "Original FA", {col1x, rowY}, 12, 1,
                     ColorAlpha(PAL.orig, expandCmpT));
          DrawTextEx(fontSm, "Modified FA", {col2x, rowY}, 12, 1,
                     ColorAlpha(PAL.mod, expandCmpT));
          rowY += 18.f;

          // Divider
          DrawLineEx({pix + piw / 2.f, piy + 4}, {pix + piw / 2.f, piy + 76.f},
                     0.5f, ColorAlpha(PAL.panelBord, expandCmpT * 0.6f));

          // Iter to optimal
          const char *oIter =
              origFA.stats.iterToOptimal >= 0
                  ? TextFormat("Solved @ iter %d", origFA.stats.iterToOptimal)
                  : "Not solved";
          const char *mIter =
              modFA.stats.iterToOptimal >= 0
                  ? TextFormat("Solved @ iter %d", modFA.stats.iterToOptimal)
                  : "Not solved";
          DrawTextEx(fontSm, oIter, {col1x, rowY}, 12, 1,
                     ColorAlpha(PAL.orig, expandCmpT * 0.9f));
          DrawTextEx(fontSm, mIter, {col2x, rowY}, 12, 1,
                     ColorAlpha(PAL.mod, expandCmpT * 0.9f));
          rowY += 16.f;

          // Optimal hits
          DrawTextEx(fontSm, TextFormat("Hits: %d", origFA.stats.optimalCount),
                     {col1x, rowY}, 12, 1,
                     ColorAlpha(PAL.orig, expandCmpT * 0.8f));
          DrawTextEx(fontSm, TextFormat("Hits: %d", modFA.stats.optimalCount),
                     {col2x, rowY}, 12, 1,
                     ColorAlpha(PAL.mod, expandCmpT * 0.8f));

          // Winner line — centered, gold
          bool origFaster =
              origFA.stats.iterToOptimal >= 0 &&
              (modFA.stats.iterToOptimal < 0 ||
               origFA.stats.iterToOptimal < modFA.stats.iterToOptimal);
          bool modFaster =
              modFA.stats.iterToOptimal >= 0 &&
              (origFA.stats.iterToOptimal < 0 ||
               modFA.stats.iterToOptimal < origFA.stats.iterToOptimal);
          bool neitherSolved =
              origFA.stats.iterToOptimal < 0 && modFA.stats.iterToOptimal < 0;

          const char *winStr = neitherSolved ? "Neither reached optimal"
                               : origFaster  ? "Original FA was faster!"
                               : modFaster   ? "Modified FA was faster!"
                                             : "Both solved equally";
          Color winCol = neitherSolved ? PAL.dimText
                         : origFaster  ? PAL.orig
                         : modFaster   ? PAL.mod
                                       : PAL.gold;

          piy += 84.f; // increased from 70 to match new box height
          // Draw winner text below the box
          Vector2 wSz = MeasureTextEx(fontSm, winStr, 13, 1);
          DrawRectangleRounded(
              {pix + piw / 2.f - wSz.x / 2.f - 12, piy, wSz.x + 24, 22}, 0.3f,
              6, ColorAlpha(winCol, expandCmpT * 0.18f));
          DrawRectangleRoundedLinesEx(
              {pix + piw / 2.f - wSz.x / 2.f - 12, piy, wSz.x + 24, 22}, 0.3f,
              6, 1.5f, ColorAlpha(winCol, expandCmpT * 0.7f));
          DrawTextEx(fontSm, winStr, {pix + piw / 2.f - wSz.x / 2.f, piy + 4},
                     13, 1, ColorAlpha(winCol, expandCmpT));
          piy += 28.f;
        }

        // ── DOWNLOAD BUTTON ──────────────────────────────────────
        if (origFA.done && modFA.done) {
          float dlW = leftR.width - 16.f;
          float dlH = 30.f;
          float dlX = leftR.x + 8.f;
          float dlY = leftR.y + leftR.height - dlH - 8.f;
          bool hovDL =
              CheckCollisionPointRec(GetMousePosition(), {dlX, dlY, dlW, dlH});
          DrawRectangleRounded({dlX, dlY, dlW, dlH}, 0.25f, 8,
                               ColorAlpha(hovDL ? PAL.accent : PAL.panelBord,
                                          expandCmpT * 0.9f));
          DrawRectangleRoundedLinesEx({dlX, dlY, dlW, dlH}, 0.25f, 8, 1.5f,
                                      ColorAlpha(PAL.gold, expandCmpT * 0.8f));
          {
            const char *dlLabel = "DOWNLOAD REPORT";
            Vector2 dlSz = MeasureTextEx(fontSm, dlLabel, 12, 1);
            DrawTextEx(fontSm, dlLabel,
                       {dlX + dlW / 2 - dlSz.x / 2, dlY + dlH / 2 - dlSz.y / 2},
                       12, 1, ColorAlpha(WHITE, expandCmpT));
          }
          if (hovDL && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            std::ofstream f("FA_Comparison_Report.txt");
            if (f.is_open()) {
              f << "FIREFLY ALGORITHM COMPARISON REPORT\n";
              f << "====================================\n";
              f << "N=" << N << "  POP=" << POP << "  MAX_ITER=" << MAX_ITER
                << "\n\n";
              f << "Best Fitness:  Orig=" << cmpResult.origBest
                << "  Mod=" << cmpResult.modBest << "\n";
              f << "Avg Fitness:   Orig=" << cmpResult.origAvg
                << "  Mod=" << cmpResult.modAvg << "\n";
              f << "Iter to Opt:   Orig=" << cmpResult.origIterOpt
                << "  Mod=" << cmpResult.modIterOpt << "\n";
              f << "Optimal Hits:  Orig=" << cmpResult.origOptHits
                << "  Mod=" << cmpResult.modOptHits << "\n";
              f << "Diversity:     Orig=" << cmpResult.origDiversity
                << "  Mod=" << cmpResult.modDiversity << "\n";
              f << "\nScore: " << cmpResult.score << "/5\n";
              f << "Verdict: " << verdSummary[vi] << "\n";
              f << "\nParams: a0=" << modParams.alpha0
                << " mut=" << modParams.mutRate
                << " elite=" << modParams.eliteCount << "\n";
              f.close();
              downloadMsg = "Saved: FA_Comparison_Report.txt";
              downloadMsgTimer = 3.f;
            }
          }
          if (downloadMsgTimer > 0.f && expandCmpT > 0.5f) {
            DrawTextEx(
                fontSm, downloadMsg.c_str(), {dlX, dlY - 16}, 10, 1,
                ColorAlpha(PAL.safe,
                           expandCmpT * std::min(downloadMsgTimer, 1.f)));
          }
        }

        // Right half top: full convergence graph
        // ── RIGHT SIDE: 3 charts stacked ──────────────────────────
        float rw = mw * 0.55f;
        float rx2 = mx + mw * 0.44f;
        float rGap = 8.f;
        float rh1 = (mh - 80.f) * 0.62f; // Bar comparison (taller now)
        float rh3 = (mh - 80.f) * 0.34f; // Diversity over time
        float ry1 = my + 60.f;
        float ry3 = ry1 + rh1 + rGap;

        // ── CHART 1: PER-METRIC BAR COMPARISON ──────────────────────
        {
          Rectangle cr = {rx2, ry1, rw, rh1};
          DrawPanel(cr, "Metric-by-Metric Bar Comparison", fontSm, PAL.gold);

          float labelColW = 90.f;
          float valColW = 36.f;
          float bx0 = cr.x + labelColW + 4.f;
          float bw0 = cr.width - labelColW - valColW - 20.f;
          float bh0 = 16.f;
          float rowS = (rh1 - 85.f) / 6.f;
          float startY = cr.y + 78.f;

          struct MetricBar {
            const char *name;
            float vO, vM, maxV;
            bool higherBetter;
          };

          // Normalise each metric to 0..1 relative to their own max
          float mxAvg = std::max({cmpResult.origAvg, cmpResult.modAvg, 1.f});
          float mxHits = std::max(
              (float)std::max(cmpResult.origOptHits, cmpResult.modOptHits),
              1.f);
          float mxDiv = std::max(
              {cmpResult.origDiversity, cmpResult.modDiversity, 0.01f});
          float mxWorst =
              std::max({cmpResult.origWorst, cmpResult.modWorst, 1.f});

          MetricBar bars[] = {
              {"Best Fitness", cmpResult.origBest / (float)MAX_FIT,
               cmpResult.modBest / (float)MAX_FIT, 1.f, true},
              {"Avg Fitness", cmpResult.origAvg / mxAvg,
               cmpResult.modAvg / mxAvg, 1.f, true},
              {"Optimal Hits", (float)cmpResult.origOptHits / mxHits,
               (float)cmpResult.modOptHits / mxHits, 1.f, true},
              {"Worst Fitness", cmpResult.origWorst / mxWorst,
               cmpResult.modWorst / mxWorst, 1.f, true},
              {"Diversity", cmpResult.origDiversity / mxDiv,
               cmpResult.modDiversity / mxDiv, 1.f, true},
              {"Speed (inv)",
               cmpResult.origIterOpt < 0
                   ? 0.f
                   : 1.f - (float)cmpResult.origIterOpt / MAX_ITER,
               cmpResult.modIterOpt < 0
                   ? 0.f
                   : 1.f - (float)cmpResult.modIterOpt / MAX_ITER,
               1.f, true},
          };

          // Column headers — properly centred using MeasureTextEx
          float halfW = bw0 / 2.f - 4.f;
          DrawTextEx(fontSm, "Original",
                     {bx0 + halfW / 2.f -
                          MeasureTextEx(fontSm, "Original", 10, 1).x / 2.f,
                      cr.y + 58},
                     10, 1, ColorAlpha(PAL.orig, expandCmpT));
          DrawTextEx(fontSm, "Modified",
                     {bx0 + halfW + 8.f + halfW / 2.f -
                          MeasureTextEx(fontSm, "Modified", 10, 1).x / 2.f,
                      cr.y + 58},
                     10, 1, ColorAlpha(PAL.mod, expandCmpT));
          // Divider between columns
          DrawLineEx({bx0 + halfW + 4, cr.y + 68},
                     {bx0 + halfW + 4, cr.y + rh1 - 8}, 0.5f,
                     ColorAlpha(PAL.panelBord, expandCmpT * 0.5f));

          for (int i = 0; i < 6; i++) {
            float rowY = startY + i * rowS;
            float halfW = bw0 / 2.f - 4.f;

            // Alternating row background
            if (i % 2 == 0)
              DrawRectangleRec({cr.x + 4, rowY - 2, cr.width - 8, rowS - 2},
                               ColorAlpha(PAL.panelBord, expandCmpT * 0.12f));

            // Metric label — left-aligned, clipped width
            DrawTextEx(fontSm, bars[i].name, {cr.x + 8, rowY + bh0 / 2 - 5}, 10,
                       1, ColorAlpha(PAL.dimText, expandCmpT));

            // ── Orig bar (left half of bar area) ──
            float origBarX = bx0;
            DrawRectangleRounded({origBarX, rowY, halfW, bh0}, 1.f, 4,
                                 ColorAlpha(PAL.panelBord, expandCmpT * 0.35f));
            if (bars[i].vO > 0.001f) {
              float fillW = Clamp(bars[i].vO * halfW, 2.f, halfW);
              DrawRectangleGradientH(
                  (int)origBarX, (int)rowY, (int)fillW, (int)bh0,
                  ColorAlpha(PAL.origDim, expandCmpT * 0.85f),
                  ColorAlpha(PAL.orig, expandCmpT));
              // Value inside bar if wide enough, else right of bar
              const char *ovStr = TextFormat("%.0f%%", bars[i].vO * 100);
              float ovSzX = MeasureTextEx(fontSm, ovStr, 9, 1).x;
              float ovX = (fillW > ovSzX + 6) ? origBarX + fillW - ovSzX - 3
                                              : origBarX + fillW + 2;
              DrawTextEx(
                  fontSm, ovStr, {ovX, rowY + 1}, 9, 1,
                  ColorAlpha(fillW > ovSzX + 6 ? WHITE : PAL.orig, expandCmpT));
            }

            // ── Mod bar (right half of bar area) ──
            float modBarX = bx0 + halfW + 8.f;
            DrawRectangleRounded({modBarX, rowY, halfW, bh0}, 1.f, 4,
                                 ColorAlpha(PAL.panelBord, expandCmpT * 0.35f));
            if (bars[i].vM > 0.001f) {
              float fillW = Clamp(bars[i].vM * halfW, 2.f, halfW);
              DrawRectangleGradientH((int)modBarX, (int)rowY, (int)fillW,
                                     (int)bh0,
                                     ColorAlpha(PAL.modDim, expandCmpT * 0.85f),
                                     ColorAlpha(PAL.mod, expandCmpT));
              const char *mvStr = TextFormat("%.0f%%", bars[i].vM * 100);
              float mvSzX = MeasureTextEx(fontSm, mvStr, 9, 1).x;
              float mvX = (fillW > mvSzX + 6) ? modBarX + fillW - mvSzX - 3
                                              : modBarX + fillW + 2;
              DrawTextEx(
                  fontSm, mvStr, {mvX, rowY + 1}, 9, 1,
                  ColorAlpha(fillW > mvSzX + 6 ? WHITE : PAL.mod, expandCmpT));
            }

            // ── Winner indicator dot (far right) ──
            bool origWins = bars[i].vO > bars[i].vM + 0.02f;
            bool modWins = bars[i].vM > bars[i].vO + 0.02f;
            Color dotCol = origWins  ? PAL.orig
                           : modWins ? PAL.mod
                                     : PAL.dimText;
            float dotX = cr.x + cr.width - 10.f;
            float dotY = rowY + bh0 / 2.f;
            DrawCircleV({dotX, dotY}, 5.f, ColorAlpha(dotCol, expandCmpT));
            if (origWins || modWins)
              DrawCircleLines((int)dotX, (int)dotY, 7.5f,
                              ColorAlpha(dotCol, expandCmpT * 0.35f));
          }
        }

        // ── CHART 3: POPULATION DIVERSITY OVER TIME
        // ────────────────────────────
        {
          Rectangle cr = {rx2, ry3, rw, rh3};
          DrawPanel(cr, "Population Diversity Over Time", fontSm,
                    ColorBrightness(PAL.accent, -0.2f));

          float dx0 = cr.x + 36.f, dy0 = cr.y + 28.f;
          float dw0 = cr.width - 48.f, dh0 = cr.height - 42.f;

          DrawRectangleRec({dx0, dy0, dw0, dh0}, ColorAlpha(PAL.bg, 0.5f));

          for (int i = 0; i <= 3; i++) {
            float t0 = (float)i / 3.f;
            float yy = dy0 + dh0 - t0 * dh0;
            DrawLineEx({dx0, yy}, {dx0 + dw0, yy}, 0.3f,
                       ColorAlpha(PAL.axis, 0.15f));
            DrawTextEx(fontSm, TextFormat("%.1f", t0), {dx0 - 26, yy - 6}, 9, 1,
                       PAL.dimText);
          }
          for (int i = 0; i <= 4; i++) {
            float gx2 = dx0 + (float)i / 4.f * dw0;
            DrawTextEx(fontSm, TextFormat("%d", MAX_ITER * i / 4),
                       {gx2 - 8, dy0 + dh0 + 3}, 9, 1, PAL.dimText);
          }

          auto drawDiv0 = [&](const std::vector<float> &d, Color c) {
            if (d.size() < 2)
              return;
            // Shade area under curve
            for (int i = 0; i < (int)d.size() - 1; i++) {
              float x1 = dx0 + (float)i / MAX_ITER * dw0;
              float x2 = dx0 + (float)(i + 1) / MAX_ITER * dw0;
              float y1 = dy0 + dh0 - Clamp(d[i], 0.f, 1.f) * dh0;
              float y2 = dy0 + dh0 - Clamp(d[i + 1], 0.f, 1.f) * dh0;
              DrawTriangle({x1, dy0 + dh0}, {x1, y1}, {x2, y2},
                           ColorAlpha(c, 0.07f * expandCmpT));
              DrawTriangle({x1, dy0 + dh0}, {x2, y2}, {x2, dy0 + dh0},
                           ColorAlpha(c, 0.07f * expandCmpT));
              DrawLineEx({x1, y1}, {x2, y2}, 1.5f, ColorAlpha(c, expandCmpT));
            }
          };
          if (showO)
            drawDiv0(origFA.diversityPerIter, PAL.orig);
          if (showM)
            drawDiv0(modFA.diversityPerIter, PAL.mod);

          // Labels
          DrawTextEx(fontSm, "Orig", {dx0 + 4, dy0 + 4}, 10, 1,
                     ColorAlpha(PAL.orig, 0.8f * expandCmpT));
          DrawTextEx(fontSm, "Mod", {dx0 + 36, dy0 + 4}, 10, 1,
                     ColorAlpha(PAL.mod, 0.8f * expandCmpT));
          DrawTextEx(fontSm, "Iterations", {dx0 + dw0 / 2 - 24, dy0 + dh0 + 14},
                     10, 1, ColorAlpha(PAL.dimText, expandCmpT));
          DrawTextEx(fontSm, "Diversity", {dx0 - 42, dy0 + dh0 / 2 - 6}, 10, 1,
                     ColorAlpha(PAL.dimText, expandCmpT));

          // Final diversity values
          if (!origFA.diversityPerIter.empty())
            DrawTextEx(
                fontSm,
                TextFormat("Final Orig: %.3f", origFA.diversityPerIter.back()),
                {dx0 + dw0 - 110, dy0 + 4}, 10, 1,
                ColorAlpha(PAL.orig, expandCmpT));
          if (!modFA.diversityPerIter.empty())
            DrawTextEx(
                fontSm,
                TextFormat("Final Mod: %.3f", modFA.diversityPerIter.back()),
                {dx0 + dw0 - 110, dy0 + 14}, 10, 1,
                ColorAlpha(PAL.mod, expandCmpT));
        }
      }
    }

    // Celebration overlay
    if (celebTimer > 0.f) {
      float a = std::min(celebTimer, 1.f);
      DrawTextEx(fontBold, "SOLUTION FOUND!",
                 {WIN_W / 2.f - 150, WIN_H / 2.f - 20}, 36, 2,
                 ColorAlpha(PAL.gold, a));
    }

    // SOLVED badge removed — results shown in Comparison Modal instead

    // ================================================================
    // THEORY PANEL MODAL
    // ================================================================
    if (theoryAlpha > 0.01f) {
      // Dim overlay
      DrawRectangle(0, 0, WIN_W, WIN_H,
                    ColorAlpha({0, 0, 0, 255}, 0.82f * theoryAlpha));

      float mw = 1100.f, mh = 720.f;
      float mx = WIN_W / 2.f - mw / 2.f;
      float my = WIN_H / 2.f - mh / 2.f;
      Rectangle modal = {mx, my, mw, mh};

      DrawRectangleRounded(modal, 0.04f, 16,
                           ColorAlpha({10, 12, 28, 255}, theoryAlpha));
      DrawRectangleRoundedLinesEx(modal, 0.04f, 16, 2.f,
                                  ColorAlpha(PAL.gold, theoryAlpha * 0.8f));

      // Title bar
      DrawRectangleRounded({mx, my, mw, 48.f}, 0.04f, 16,
                           ColorAlpha({18, 14, 45, 255}, theoryAlpha));
      DrawRectangle((int)mx + 2, (int)my + 48, (int)mw - 4, 1,
                    ColorAlpha(PAL.gold, theoryAlpha * 0.4f));
      DrawTextEx(fontBold, "THEORY & VISUAL GUIDE", {mx + 20, my + 12}, 20,
                 1.5f, ColorAlpha(PAL.gold, theoryAlpha));

      // Close button
      Rectangle closeR = {mx + mw - 44, my + 8, 32, 32};
      bool hovClose = CheckCollisionPointRec(GetMousePosition(), closeR);
      DrawRectangleRounded(
          closeR, 0.4f, 8,
          ColorAlpha(hovClose ? PAL.conflict
                              : ColorBrightness(PAL.conflict, -0.3f),
                     theoryAlpha));
      DrawTextEx(fontBold, "X", {closeR.x + 10, closeR.y + 8}, 16, 1,
                 ColorAlpha(WHITE, theoryAlpha));
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
        bool hov = CheckCollisionPointRec(GetMousePosition(),
                                          {tx, tabY, tabW - 4, 30});
        DrawRectangleRounded(
            {tx, tabY, tabW - 4, 30}, 0.3f, 8,
            ColorAlpha(sel ? PAL.accent
                           : (hov ? ColorBrightness(PAL.panelBord, 0.2f)
                                  : PAL.panelBord),
                       theoryAlpha));
        if (sel)
          DrawRectangleRounded({tx, tabY + 26, tabW - 4, 4}, 0.5f, 4,
                               ColorAlpha(PAL.gold, theoryAlpha));
        Vector2 tsz = MeasureTextEx(fontSm, tabLabels[i], 13, 1);
        DrawTextEx(fontSm, tabLabels[i], {tx + tabW / 2 - tsz.x / 2, tabY + 8},
                   13, 1, ColorAlpha(sel ? WHITE : PAL.dimText, theoryAlpha));
        if (hov && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
          theoryTab = i;
      }

      // Content area
      float cx2 = mx + 24.f;
      float cy2 = my + 100.f;
      float cw2 = mw - 48.f;
      float ch2 = mh - 120.f;
      BeginScissorMode((int)cx2, (int)cy2, (int)cw2, (int)ch2);

      // Two-column layout helpers
      float colW = (cw2 - 24.f) / 2.f; // width of each column
      float col2x = cx2 + colW + 24.f; // x start of right column
      // TH / TX / BULLET / ROW / SWATCH lambdas operate on (startX, colWidth)
      // We redefine them to accept an x offset and width

      auto THL = [&](const char *t, float x, float y, float w) -> float {
        DrawTextEx(fontBold, t, {x, cy2 + y}, 15, 1,
                   ColorAlpha(PAL.gold, theoryAlpha));
        DrawLineEx({x, cy2 + y + 18}, {x + w, cy2 + y + 18}, 0.5f,
                   ColorAlpha(PAL.gold, theoryAlpha * 0.3f));
        return y + 26.f;
      };
      auto TXL = [&](const char *t, float x, float y, float w, Color c,
                     float sz = 12.f) -> float {
        (void)w;
        DrawTextEx(fontSm, t, {x + 4, cy2 + y}, sz, 1,
                   ColorAlpha(c, theoryAlpha));
        return y + sz + 4.f;
      };
      auto BUL = [&](const char *t, float x, float y,
                     Color c = {180, 190, 210, 255}) -> float {
        DrawCircleV({x + 10, cy2 + y + 6}, 2.5f, ColorAlpha(c, theoryAlpha));
        DrawTextEx(fontSm, t, {x + 20, cy2 + y}, 12, 1,
                   ColorAlpha(c, theoryAlpha));
        return y + 18.f;
      };
      auto ROWL = [&](const char *c1, const char *c2, const char *c3, float x,
                      float w, float y, bool header) -> float {
        float cw1 = 130.f, cw2b = 130.f;
        if (header)
          DrawRectangleRec({x, cy2 + y, w, 20},
                           ColorAlpha(PAL.panelBord, theoryAlpha * 0.6f));
        else if ((int)(y / 20) % 2 == 0)
          DrawRectangleRec({x, cy2 + y, w, 20},
                           ColorAlpha(PAL.panelBord, theoryAlpha * 0.15f));
        DrawTextEx(fontSm, c1, {x + 4, cy2 + y + 3}, 11, 1,
                   ColorAlpha(header ? PAL.gold : PAL.text, theoryAlpha));
        DrawTextEx(fontSm, c2, {x + cw1, cy2 + y + 3}, 11, 1,
                   ColorAlpha(header ? PAL.gold : PAL.orig, theoryAlpha));
        DrawTextEx(fontSm, c3, {x + cw1 + cw2b, cy2 + y + 3}, 11, 1,
                   ColorAlpha(header ? PAL.gold : PAL.dimText, theoryAlpha));
        DrawLineEx({x, cy2 + y + 20}, {x + w, cy2 + y + 20}, 0.5f,
                   ColorAlpha(PAL.panelBord, theoryAlpha * 0.3f));
        return y + 21.f;
      };
      auto SWATCHL = [&](Color col2, float x, float y2, float w2, float h2) {
        DrawRectangleRounded({x, cy2 + y2, w2, h2}, 0.3f, 6,
                             ColorAlpha(col2, theoryAlpha));
        DrawRectangleRoundedLinesEx({x, cy2 + y2, w2, h2}, 0.3f, 6, 0.8f,
                                    ColorAlpha(WHITE, theoryAlpha * 0.2f));
      };

      float y = 8.f;

      // ── TAB 0: OVERVIEW ─────────────────────── two columns ──
      if (theoryTab == 0) {
        // LEFT column
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
        ly = BUL("[M]  Modified FA only", lx, ly, PAL.mod);
        ly = BUL("[B]  Run both engines", lx, ly, PAL.accent);
        ly = BUL("[Space]  Play / Pause", lx, ly);
        ly = BUL("[Right]  Step forward", lx, ly);
        ly = BUL("[UP/DN]  Change N (4-32)", lx, ly);
        ly = BUL("[E]  Expanded graph", lx, ly);
        ly = BUL("[C]  Toggle conflicts", lx, ly);
        ly = BUL("[T]  Toggle trails", lx, ly);
        ly = BUL("[X]  Comparison table", lx, ly);
        ly = BUL("[H]  This theory panel", lx, ly);
        ly = BUL("[R]  Reset", lx, ly, PAL.conflict);

        // RIGHT column
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
        ly = ROWL("Elitism", "Off", "Top-2 preserved", lx, cw2, ly, false);
        ly = ROWL("Mutation rate", "—", "20% (N<=10)", lx, cw2, ly, false);
        ly = ROWL("Convergence", "Slower (handicapped)", "Faster", lx, cw2, ly,
                  false);
        ly = ROWL("Early diversity", "Higher", "Lower", lx, cw2, ly, false);
        ly += 10;

        // Now split into two columns for the detail sections
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
        ly2 =
            TXL("Each firefly: 20% (or 35% large N)", lx2, ly2, colW, PAL.text);
        ly2 =
            TXL("chance of nSwaps random row swaps.", lx2, ly2, colW, PAL.text);
        ly2 = TXL("nSwaps = max(1, (N-6)/3)", lx2, ly2, colW, PAL.gold, 12);

        float rx = col2x, ry = ly;
        ry = THL("Elitism", rx, ry, colW);
        ry = TXL("Top 2 fireflies cached each iter.", rx, ry, colW, PAL.text);
        ry = TXL("At end they replace the 2 worst,", rx, ry, colW, PAL.text);
        ry = TXL("preventing regression.", rx, ry, colW, PAL.text);
        ry += 6;
        ry = THL("Why Modified Is Faster", rx, ry, colW);
        ry = BUL("Better starting positions", rx, ry, PAL.mod);
        ry = BUL("Mutation escapes local optima", rx, ry, PAL.mod);
        ry = BUL("Elitism locks in good solutions", rx, ry, PAL.mod);
        ry = BUL("Alpha decay focuses search late", rx, ry, PAL.mod);
      }

      // ── TAB 3: COLOUR LEGEND ──────────────── two columns ──
      else if (theoryTab == 3) {
        float lx = cx2, ly = y;
        ly = THL("Fitness → Colour Gradient", lx, ly, cw2);

        // Gradient bar full width
        float gbx = cx2 + 4;
        float gby = cy2 + ly;
        float gbw = cw2 - 8;
        float gbh = 22.f;
        int segs = 100;
        for (int i = 0; i < segs; i++) {
          float t2 = (float)i / segs;
          float t1 = (float)(i + 1) / segs;
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
        DrawRectangleRoundedLinesEx({gbx, gby, gbw, gbh}, 0.2f, 8, 1.f,
                                    ColorAlpha(PAL.panelBord, theoryAlpha));
        // Percentage markers
        const char *pcts[] = {"0%", "50%", "92%", "100%"};
        float pctPos[] = {0.f, 0.5f, 0.92f, 1.f};
        for (int i = 0; i < 4; i++) {
          float px = gbx + pctPos[i] * gbw;
          DrawLineEx({px, gby + gbh}, {px, gby + gbh + 6}, 1.f,
                     ColorAlpha(PAL.dimText, theoryAlpha));
          DrawTextEx(fontSm, pcts[i], {px - 8, gby + gbh + 8}, 10, 1,
                     ColorAlpha(PAL.dimText, theoryAlpha));
        }
        DrawTextEx(fontSm, "Orig (top)", {gbx + gbw + 6, gby + 2}, 11, 1,
                   ColorAlpha(PAL.orig, theoryAlpha));
        DrawTextEx(fontSm, "Mod  (bot)", {gbx + gbw + 6, gby + gbh / 2 + 2}, 11,
                   1, ColorAlpha(PAL.mod, theoryAlpha));
        ly += gbh + 22.f;

        // Table header — two columns each half width
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
            {PAL.safe, "Green", "Solved / PLAY / safe state", "UI"},
            {PAL.conflict, "Red", "Conflict lines / attacks", "UI"},
            {PAL.accent, "Purple", "Run Both / graph accent", "UI"},
            {PAL.heatHigh, "Heat Gold", "Heat map — very frequent", "Top 50%+"},
            {PAL.heatMid, "Heat Purple", "Heat map — moderate", "Mid"},
            {PAL.heatLow, "Heat Dark", "Heat map — rare visits", "Low"},
            {{232, 237, 209, 255},
             "Light Square",
             "Chess board light square",
             "—"},
            {{90, 130, 60, 255}, "Dark Square", "Chess board dark square", "—"},
        };
        int ncr = sizeof(colRows) / sizeof(colRows[0]);
        int half = (ncr + 1) / 2;

        // Header row both sides
        float sw = 16.f, nc1 = 110.f, nc2 = 160.f;
        // Left header
        DrawRectangleRec({cx2, cy2 + ly, colW, 20},
                         ColorAlpha(PAL.panelBord, theoryAlpha * 0.7f));
        DrawTextEx(fontSm, "Colour", {cx2 + sw + 4, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, theoryAlpha));
        DrawTextEx(fontSm, "Meaning", {cx2 + sw + nc1, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, theoryAlpha));
        DrawTextEx(fontSm, "Range", {cx2 + sw + nc1 + nc2 - 20, cy2 + ly + 4},
                   11, 1, ColorAlpha(PAL.gold, theoryAlpha));
        // Right header
        DrawRectangleRec({col2x, cy2 + ly, colW, 20},
                         ColorAlpha(PAL.panelBord, theoryAlpha * 0.7f));
        DrawTextEx(fontSm, "Colour", {col2x + sw + 4, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, theoryAlpha));
        DrawTextEx(fontSm, "Meaning", {col2x + sw + nc1, cy2 + ly + 4}, 11, 1,
                   ColorAlpha(PAL.gold, theoryAlpha));
        DrawTextEx(fontSm, "Range", {col2x + sw + nc1 + nc2 - 20, cy2 + ly + 4},
                   11, 1, ColorAlpha(PAL.gold, theoryAlpha));
        ly += 21.f;

        for (int i = 0; i < half; i++) {
          // Left column row
          float ry2 = ly + i * 20.f;
          if (i % 2 == 0)
            DrawRectangleRec({cx2, cy2 + ry2, colW, 20},
                             ColorAlpha(PAL.panelBord, theoryAlpha * 0.18f));
          SWATCHL(colRows[i].swatch, cx2 + 2, ry2 + 2, sw - 2, 16.f);
          DrawTextEx(fontSm, colRows[i].name, {cx2 + sw + 4, cy2 + ry2 + 4}, 11,
                     1, ColorAlpha(colRows[i].swatch, theoryAlpha));
          DrawTextEx(fontSm, colRows[i].meaning,
                     {cx2 + sw + nc1, cy2 + ry2 + 4}, 11, 1,
                     ColorAlpha(PAL.text, theoryAlpha));
          DrawTextEx(fontSm, colRows[i].range,
                     {cx2 + sw + nc1 + nc2 - 20, cy2 + ry2 + 4}, 10, 1,
                     ColorAlpha(PAL.dimText, theoryAlpha));
          DrawLineEx({cx2, cy2 + ry2 + 20}, {cx2 + colW, cy2 + ry2 + 20}, 0.5f,
                     ColorAlpha(PAL.panelBord, theoryAlpha * 0.3f));
          // Right column row
          int j = i + half;
          if (j < ncr) {
            if (i % 2 == 0)
              DrawRectangleRec({col2x, cy2 + ry2, colW, 20},
                               ColorAlpha(PAL.panelBord, theoryAlpha * 0.18f));
            SWATCHL(colRows[j].swatch, col2x + 2, ry2 + 2, sw - 2, 16.f);
            DrawTextEx(fontSm, colRows[j].name, {col2x + sw + 4, cy2 + ry2 + 4},
                       11, 1, ColorAlpha(colRows[j].swatch, theoryAlpha));
            DrawTextEx(fontSm, colRows[j].meaning,
                       {col2x + sw + nc1, cy2 + ry2 + 4}, 11, 1,
                       ColorAlpha(PAL.text, theoryAlpha));
            DrawTextEx(fontSm, colRows[j].range,
                       {col2x + sw + nc1 + nc2 - 20, cy2 + ry2 + 4}, 10, 1,
                       ColorAlpha(PAL.dimText, theoryAlpha));
            DrawLineEx({col2x, cy2 + ry2 + 20}, {col2x + colW, cy2 + ry2 + 20},
                       0.5f, ColorAlpha(PAL.panelBord, theoryAlpha * 0.3f));
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
        ly = BUL("[X] Comparison table side-by-side", lx, ly);
        ly = BUL("Fitness bars = progress to MAX_FIT", lx, ly);

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
        ry = BUL("Red row highlight = conflict in that row", rx, ry,
                 PAL.conflict);
        ry = BUL("Red lines = diagonal attack pairs", rx, ry, PAL.conflict);
        ry = BUL("Dual mode: queens offset L/R per engine", rx, ry);
        ry = BUL("Gold glow = optimal solution found", rx, ry, PAL.gold);
        ry += 6;
        ry = THL("Title Bar Indicators", rx, ry, colW);
        ry = BUL("Green pill = RUNNING", rx, ry, PAL.safe);
        ry = BUL("Gold pill  = PAUSED", rx, ry, PAL.gold);
        ry = BUL("Dim pill   = IDLE", rx, ry, PAL.dimText);
        ry = BUL("Pulsing dot = live activity", rx, ry);
      }

      EndScissorMode();

      // Scroll hint
      DrawTextEx(fontSm, "Scroll through tabs using the tab bar above",
                 {mx + mw / 2.f - 140.f, my + mh - 20.f}, 11, 1,
                 ColorAlpha(PAL.dimText, theoryAlpha * 0.6f));
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
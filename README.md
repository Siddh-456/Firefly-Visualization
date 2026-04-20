<div align="center">
  <h1>Enhanced Firefly Algorithm for N-Queens</h1>
  <p><strong>A C++17 desktop visualizer solving the N-Queens problem with a modified Firefly Algorithm — built for faster convergence, better stability, and rich real-time analysis.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++17" />
    <img src="https://img.shields.io/badge/Raylib-5.x-111111?style=for-the-badge" alt="Raylib 5.x" />
    <img src="https://img.shields.io/badge/OpenGL-Graphics-5586A4?style=for-the-badge&logo=opengl" alt="OpenGL" />
    <img src="https://img.shields.io/badge/SFML-Audio-8CC445?style=for-the-badge" alt="SFML" />
    <img src="https://img.shields.io/badge/Problem-N--Queens-8A2BE2?style=for-the-badge" alt="N-Queens" />
    <img src="https://img.shields.io/badge/Focus-Swarm%20Optimization-0A7E8C?style=for-the-badge" alt="Swarm Optimization" />
    <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-3A3A3A?style=for-the-badge" alt="Windows and Linux" />
  </p>
</div>

<p align="center">
  <img src="images/visualizer_main.png" alt="Main visualizer" width="100%" />
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#built-with">Built With</a> |
  <a href="#algorithm-design">Algorithm Design</a> |
  <a href="#visual-gallery">Visual Gallery</a> |
  <a href="#results">Results</a> |
  <a href="#run-on-your-pc">Run on Your PC</a>
</p>

---

## Overview

This project solves the N-Queens problem using an enhanced Firefly Algorithm adapted for a discrete permutation search space. Instead of treating the problem like a generic continuous optimizer, the implementation redesigns the movement, repair, mutation, and initialization stages so the swarm can work on valid chessboard configurations directly.

The result is a visually rich desktop application that lets you watch the swarm evolve in real time, compare the original and modified Firefly Algorithm side by side, inspect convergence behavior, tune parameters live, and hear audio feedback tied to algorithm events.

## Built With

| Layer | Tools and Technologies |
| --- | --- |
| Language | C++17 |
| Graphics | Raylib 5.x with OpenGL backend |
| Rendering | OpenGL (via Raylib) for hardware-accelerated 2D/3D rendering |
| Audio | SFML Audio for procedural sound effects and event-driven feedback |
| Algorithm | Custom Firefly Algorithm (swarm metaheuristic, discrete permutation space) |
| Build | g++ / MinGW (Windows), g++ (Linux) |
| Platform | Windows, Linux |

## Why This Version Is Better

- Uses permutation-based board encoding so each solution always represents one queen per row and one queen per column.
- Tracks fitness through non-attacking queen pairs, focusing conflict analysis on diagonals only.
- Adds adaptive randomness so the search explores early and sharpens later.
- Uses swap-based mutation to increase diversity without breaking permutation validity.
- Seeds part of the population with heuristic initialization for stronger starting states.
- Preserves top candidates with elitism and restores diversity through stagnation handling.
- Visualizes convergence, conflicts, heatmaps, and comparison metrics in one interface.
- Plays event-driven audio feedback on convergence, stagnation, and conflict resolution via SFML.

## Algorithm Design

| Component | Implementation |
| --- | --- |
| Solution model | A firefly is a permutation vector where index = row and value = column |
| Fitness | `maxPairs - diagonalConflicts`, where `maxPairs = N * (N - 1) / 2` |
| Attraction | Fireflies move toward brighter solutions using distance-weighted attractiveness |
| Repair | Invalid updates are repaired back into valid permutations |
| Mutation | Random swap mutation adds controlled diversity |
| Adaptation | `alpha` decays across iterations to shift from exploration to exploitation |
| Recovery | Partial reinitialization helps escape stagnation |

### Modified Firefly Upgrades

| Upgrade | What it changes |
| --- | --- |
| Adaptive randomness | Reduces alpha over time for smoother convergence |
| Heuristic initialization | Starts part of the swarm from lower-conflict configurations |
| Swap mutation | Introduces exploration while preserving permutation structure |
| Elitism | Carries the best fireflies into the next generation unchanged |
| Repair mechanism | Fixes duplicate or missing column values after movement |
| Stagnation detection | Reinjects diversity when progress stalls |

## Feature Set

- Real-time chessboard visualization of the current best solution
- Side-by-side comparison mode for original vs modified Firefly Algorithm
- Convergence graphs for best, average, and worst fitness
- Heatmap-based search-space analysis
- Conflict visualization for attacking queens
- Live parameter control for board size, mutation rate, alpha, heuristic ratio, and elitism
- Hardware-accelerated rendering via OpenGL through Raylib
- Procedural audio feedback using SFML tied to algorithm events (convergence, stagnation, conflict resolution)

## Visual Gallery

| Main interface | Parameter tuning |
| --- | --- |
| ![Main interface](images/visualizer_main.png) | ![Parameter tuning](images/param_tuning.png) |

| Original Firefly Algorithm | Modified Firefly Algorithm |
| --- | --- |
| ![Original FA for N=7](images/orig_fa_n7.png) | ![Modified FA for N=7](images/mod_fa_n7.png) |

| Heuristic initialization | Swap mutation |
| --- | --- |
| ![Heuristic initialization](images/heuristic_init.png) | ![Swap mutation](images/swap_mutation.png) |

| Original heatmap | Modified heatmap |
| --- | --- |
| ![Original heatmap](images/heatmap_orig.png) | ![Modified heatmap](images/heatmap_mod.png) |

<p align="center">
  <img src="images/comparison_report.png" alt="Comparison report" width="100%" />
</p>

## Results

The modified Firefly Algorithm consistently performs better than the baseline version in the report experiments. The main improvement comes from combining stronger initialization, adaptive exploration, elitism, and diversity-preserving mutation in a way that fits the discrete structure of the N-Queens problem.

| Criterion | Standard FA | Modified FA |
| --- | --- | --- |
| Convergence speed | Slower and more irregular | Faster and more stable |
| Solution quality | Good but inconsistent | Higher and more reliable |
| Resistance to local optima | Limited | Improved through mutation and stagnation recovery |
| Population quality | Lower average fitness | Stronger average fitness across iterations |
| Search coverage | Tends to narrow early | Explores more broadly before converging |

## Run On Your PC

### 1. Prerequisites

- C++17-compatible compiler (g++ / MinGW)
- [Raylib 5.x](https://www.raylib.com/) installed and linked
- [SFML](https://www.sfml-dev.org/) installed (for audio)
- OpenGL drivers (present on any modern system)

### 2. Clone or Download

```bash
git clone <your-repo-url>
cd DAA
```

### 3. Build

**Windows with MinGW**

```bash
g++ DAA3.cpp -o firefly_nqueens.exe -I"C:\raylib\src" -L"C:\raylib\src" -lraylib -lopengl32 -lgdi32 -lwinmm -lsfml-audio -lsfml-system -std=c++17
```

**Linux**

```bash
g++ DAA3.cpp -o firefly_nqueens -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -lsfml-audio -lsfml-system -std=c++17
```

### 4. Run

**Windows**

```bash
.\firefly_nqueens.exe
```

**Linux**

```bash
./firefly_nqueens
```

### 5. Controls

| Key | Action |
| --- | --- |
| `M` | Run modified Firefly Algorithm |
| `O` | Run original Firefly Algorithm |
| `B` | Compare both modes side by side |
| `Space` | Play or pause |
| `Right Arrow` | Single-step the simulation |
| `R` | Reset the run |
| `Up / Down` | Change N |
| `C` | Toggle conflict view |
| `T` | Toggle trails |
| `E` | Expand graph panel |
| `X` | Show or hide comparison table |
| `H` | Open theory panel |

## Project Structure

```text
DAA/
|-- DAA3.cpp
|-- nqueen.cpp
|-- DAA1.cpp
|-- DAA2.cpp
|-- DAA4.cpp
|-- images/
|   |-- visualizer_main.png
|   |-- orig_fa_n7.png
|   |-- mod_fa_n7.png
|   |-- comparison_report.png
|   |-- heatmap_orig.png
|   |-- heatmap_mod.png
|   |-- param_tuning.png
|   |-- heuristic_init.png
|   `-- swap_mutation.png
`-- README.md
```

## References

1. Xin-She Yang, *Firefly Algorithm, Stochastic Test Functions and Design Optimisation*, International Journal of Bio-Inspired Computation, 2010.
2. Raylib Documentation: https://www.raylib.com/
3. SFML Documentation: https://www.sfml-dev.org/documentation/
4. GeeksforGeeks, N-Queen Problem: https://www.geeksforgeeks.org/n-queen-problem/

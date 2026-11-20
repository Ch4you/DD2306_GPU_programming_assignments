# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Git repo for KTH DD2306 (course materials refer to it as DD2360) GPU Programming assignments. One directory per homework (`HW1/`, `HW2/`), each on its own branch (`HW2` is current; `main` is the base). There is no shared build system, test suite, or linter. Each assignment is a standalone set of CUDA `.cu` files compiled directly with `nvcc`.

The host is macOS, so `nvcc` is not installed and CUDA code cannot be built or run locally. Don't claim a kernel works without saying it was not compiled. The student runs code on a separate GPU machine (HW1 was done on Windows with an RTX 5070 Ti, compute capability 12.0).

## Layout

- `HW1/HW1_Group67_Source/` is the submitted HW1 (vector add, matrix multiply, QFT on QuEST), with a `README.md` of exact PowerShell build/profile commands. Treat it as finished reference code; its `README.md` shows the build and profiling conventions to reuse.
- `HW2/hw2-convolution.cu` is the HW2 skeleton: 1D convolution with a basic kernel and a tiled shared-memory kernel. Students fill in the `//@@ INSERT CODE HERE` markers and leave the surrounding scaffold intact.
- `HW2/HW1.txt` is an empty stray file.

## Conventions worth knowing

- Build and profile pattern from HW1: `nvcc -O3 -arch=sm_<cc> -lineinfo file.cu -o file`, then `ncu --section Occupancy --section LaunchStats` for kernels and `nsys profile --trace=cuda ...` for timelines. Set `sm_<cc>` to match the target GPU.
- Programs take problem size as `argv` (HW1 `vector_add` takes `N reps`; HW2 convolution takes `N`) and report per-stage timing (H2D, kernel, D2H).
- HW2 skeleton fixes `MASK_WIDTH 5` and the mask `[-0.25, 0.5, 1.0, 0.5, 0.25]`. `TILE_WIDTH` is left for the student to define. The skeleton uses `printf` but only includes `cuda_runtime_api.h`, `math.h`, `stdlib.h`, and `sys/time.h`, so `#include <stdio.h>` is needed to compile. Its `gpuCheck` macro only prints on error and uses `break` inside `do { } while (0)`, so it does not abort.
- Lecture slides and the course-level `CLAUDE.md` live one level up, at `../../`.

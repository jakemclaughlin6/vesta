# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Vesta?

Vesta is a pure C++ library for factor graph-based sensor fusion (SLAM / Visual SLAM), built on top of Ceres Solver. It is a ROS-agnostic fork of [fuse](https://github.com/locusrobotics/fuse) by Locus Robotics. It uses C++20, modern CMake, and the Ceres 2.2+ Manifold API.

## Build Commands

```bash
# Configure and build (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test by name
ctest --test-dir build -R test_hash_graph --output-on-failure

# Build without tests
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

### Docker

```bash
docker build -t vesta .
docker run -v $(pwd):/workspace -it vesta
```

### Formatting and Linting

```bash
# Format (uses .clang-format at repo root, Google-based, 120 char limit)
clang-format -i <file>

# Lint (uses .clang-tidy at repo root)
clang-tidy <file> -- -std=c++20 -I<include_paths>
```

## Package Architecture

Six packages, built in dependency order:

```
vesta_core  (base: Variable, Constraint, Graph, Transaction, Loss, Manifold, UUID, Timestamp)
  ├── vesta_variables  (concrete variable types: 2D/3D poses, landmarks, cameras, IMU biases)
  ├── vesta_loss       (robust loss wrappers: Huber, Cauchy, Tukey, DCS, etc.)
  └── vesta_graphs     (HashGraph: O(1) lookup, persistent ceres::Problem)
        └── vesta_constraints  (cost functions: odometry, reprojection, IMU preintegration, marginalization)
              └── vesta_optimizers  (BatchOptimizer, FixedLagSmoother)
```

Each package follows the same layout:
- `include/vesta_<pkg>/` — public headers (some have subdirs: `2d/`, `3d/`, `vision/`, `inertial/`)
- `src/` — implementation files
- `test/` — GoogleTest files named `test_*.cpp`

## Naming Conventions (enforced by .clang-tidy)

- Classes/Enums/Unions: `CamelCase`
- Functions/Methods: `camelBack`
- Variables: `lower_case`
- Protected/Private members: `lower_case_` (trailing underscore)
- Static variables: `s_lower_case` (prefix `s_`)
- Constants: `UPPER_CASE`

## Key Design Patterns

- **Synchronous, caller-driven**: No internal threads or event loops. You call `optimize()` explicitly.
- **Transaction-based**: All graph mutations go through `Transaction` objects applied atomically.
- **Boost serialization**: All variables, constraints, and graphs are serializable. New variable/constraint types must register with `BOOST_CLASS_EXPORT`.
- **UUID identity**: Each Variable and Constraint instance has a UUID. Variables combine a type hash + device UUID (+ optional timestamp). Constraints generate a random UUID.
- **Schur ordering**: `Point3DLandmark::schurGroup()` returns 0 to place landmarks in the first elimination group. Use `buildSchurOrdering()` to construct `ceres::ParameterBlockOrdering` from the graph.
- **Manifold API**: Variables define their tangent space via `ceres::Manifold` (not the deprecated LocalParameterization). `Orientation3DStamped` uses `ceres::EigenQuaternionManifold`.

## Test Helpers

Test directories contain shared fixtures/helpers:
- `vesta_core/test/`: `example_variable.h`, `example_constraint.h` — minimal concrete implementations for testing abstract interfaces
- `vesta_constraints/test/`: BAL problem data files used by `test_bal_problem`

## Dependencies

Ceres Solver (>=2.2), Eigen3, Boost (serialization), glog, SuiteSparse (CCOLAMD), GoogleTest (optional).

## Agent Guidelines

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps, or architectural decisions)
- If something goes sideways, **STOP and re-plan immediately** — do not keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Agent Team Strategy

Claude operates best with a structured team of agents, each with a distinct role. Lean on this model for any complex or multi-phase task.

|Role|Responsibility|
|---|---|
|**Orchestrator**|Owns the plan, delegates work, synthesizes results, tracks overall progress|
|**Researcher**|Handles exploration, discovery, and information gathering — offloaded to keep main context clean|
|**Implementer**|Executes a single focused task (one task per agent, no context bleed)|
|**Verifier**|Independently checks outputs, runs tests, diffs behaviour, and confirms correctness|
|**Critic**|Reviews work with a staff-engineer lens — flags hacks, suggests elegant alternatives|
|**Integrator**|Merges outputs from parallel Implementers, resolves conflicts, and ensures combined work is coherent before it reaches the Verifier|

**When to spin up a team:**

- Spawn a Researcher for any task involving exploration or uncertainty before implementation begins
- Use parallel Implementers for independent workstreams to throw more compute at hard problems
- When parallel Implementers are running, assign an Integrator to merge their outputs before verification
- Always route final outputs through a Verifier before marking complete
- Invoke the Critic on any non-trivial change to pressure-test the approach

**Key principles:**

- One task per agent — focused execution, no scope creep
- Keep the Orchestrator's context clean by offloading work aggressively
- Agents should be disposable; the Orchestrator synthesises their outputs

### 3. Self-Improvement Loop

- After ANY correction from the user, update `lessons.md` with the pattern of the mistake and the fix
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start to keep them top of mind

### 4. Verification Before Done

- Never mark a task complete without proving it works or compiles — even if it means asking the user to compile and provide the output
- Diff behaviour between main and your changes when relevant
- Ask yourself: _"Would a staff engineer approve of this?"_
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance

- For non-trivial changes, pause and ask: _"Is there a more elegant way to do this?"_
- If a fix feels hacky: _"Knowing what I know now, how would I do this starting from scratch?"_
- Strive for simplicity and maintainability, not just working code
- Skip this for small, obvious fixes — do not over-engineer
- Challenge your own work: _"If I were reviewing this PR, what feedback would I give myself?"_

---

## Task Management

1. **Plan First** — Always start with a plan, even if it's just a few bullet points. Clarify thinking and set a roadmap before touching any code or files.
2. **Verify Plan** — Before diving into implementation, review the plan to ensure it covers all steps and edge cases. Check with the user before executing if anything is ambiguous or the task is complex.
3. **Break It Down** — For complex tasks, decompose into smaller subtasks. Assign each to the appropriate agent role. This makes it easier to track progress and surface blockers early.
4. **Use Agent Teams** — If a task involves research, parallel workstreams, or validation, compose an agent team. The Orchestrator delegates; individual agents stay narrowly focused.
5. **Iterate on Feedback** — Don't push through issues. Stop, analyse, and re-plan. Route the revised plan back through the Orchestrator before resuming work.
6. **Document Lessons** — Whenever you encounter a mistake or learn something new, log it in `lessons.md`. Prevent the same mistake from happening twice.

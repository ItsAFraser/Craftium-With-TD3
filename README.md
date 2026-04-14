# Craftium-With-TD3

Craftium-With-TD3 is a course project workspace for exploring how Twin Delayed Deep Deterministic Policy Gradients (TD3) can be used with Craftium, a rich voxel-based reinforcement learning platform built on Luanti. The repository currently brings together the upstream TD3 reference implementation, a local Craftium checkout, and the scaffolding needed to reproduce experiments and build out the integration work.

## Project Goal

The goal of this project is to study whether a strong off-policy continuous-control algorithm such as TD3 can be adapted to the high-dimensional observations and game-like environments provided by Craftium. In practice, that means this repository is being used to:

- keep Craftium and TD3 code in one reproducible workspace,
- benchmark baseline behavior from the upstream implementations,
- develop task-specific training code for Craftium environments, and
- collect logs, plots, and experiment results for analysis.

## Current Repository Status

This repository is best understood as an integration workspace rather than a finished library. At the moment it includes:

- the upstream TD3 implementation in the [TD3](./TD3) directory,
- a local Craftium checkout in the [craftium](./craftium) directory,
- a setup script that clones both dependencies when they are not already present, and
- placeholder directories for results, tests, benchmarks, and analysis artifacts.

What is not yet present at the repository root is a dedicated TD3-on-Craftium training pipeline. The current training entry points still live inside the upstream projects, so the README below documents the workspace honestly and points to the relevant starting points.

## Repository Layout

```text
.
|- README.md                # Project overview and workspace guide
|- pyproject.toml           # Python project metadata and local dependency wiring
|- scripts/
|  |- setup_repo.sh         # Clones TD3 and Craftium into this workspace
|- TD3/                     # Upstream TD3 reference implementation
|- craftium/                # Local Craftium source checkout
|- analysis/                # Place for plots, notebooks, and post-run analysis
|- benchmarks/              # Benchmark-related experiments and notes
|- logs/                    # Training/runtime logs
|- results/                 # Saved outputs, metrics, model artifacts
|- tests/                   # Project-specific tests
```

## Key Components

### Craftium

Craftium provides Gymnasium-compatible 3D reinforcement learning environments implemented on top of a modified Luanti engine. The local checkout includes built-in environments such as `Craftium/ChopTree-v0`, `Craftium/OpenWorld-v0`, and `Craftium/ProcDungeons-v0`, plus example training scripts using PPO, CleanRL, Stable-Baselines3, and RLlib.

Useful files:

- [craftium/README.md](./craftium/README.md)
- [craftium/craftium-readme.md](./craftium/craftium-readme.md)
- [craftium/sb3_train.py](./craftium/sb3_train.py)
- [craftium/cleanrl_ppo_train.py](./craftium/cleanrl_ppo_train.py)

### TD3

The [TD3](./TD3) directory contains Scott Fujimoto's PyTorch reference implementation of TD3, along with DDPG comparison baselines and the original experiment driver. This code is useful as the algorithmic baseline for the course project and as a reference when adapting TD3 to Craftium-specific observations and action spaces.

Useful files:

- [TD3/README.md](./TD3/README.md)
- [TD3/main.py](./TD3/main.py)
- [TD3/TD3.py](./TD3/TD3.py)
- [TD3/utils.py](./TD3/utils.py)

## Setup

### 1. Clone or open the repository

Open this repository in VS Code or clone it locally.

### 2. Pull the external project dependencies

The workspace includes a helper script that clones the upstream repositories into the expected local folders:

```bash
./scripts/setup_repo.sh
```

By default, the script clones:

- `TD3` from `https://github.com/sfujim/TD3.git`
- `craftium` from `https://github.com/mikelma/craftium.git` with submodules

If those directories already exist as git repositories, the script leaves them untouched.

### 3. Install Python dependencies

The root [pyproject.toml](./pyproject.toml) is configured for a local editable Craftium dependency. If you use `uv`, the intended workflow is:

```bash
uv sync
```

If you prefer `pip`, install the root project and let it resolve the local Craftium dependency:

```bash
pip install -e .
```

### 4. Install Craftium system requirements

Craftium depends on Luanti and native build dependencies. On a fresh machine, Python package installation alone is not enough. If Craftium fails to build, use the documentation in the local checkout for platform-specific system setup:

- [craftium/README.md](./craftium/README.md)
- [craftium/doc/compiling/macos.md](./craftium/doc/compiling/macos.md)
- [craftium/doc/compiling/linux.md](./craftium/doc/compiling/linux.md)

## Docker Build

The root [Dockerfile](./Dockerfile) now clones Craftium during image build, so a local [craftium](./craftium) checkout is optional for Docker workflows.

Build with default Craftium source (`main` branch):

```bash
docker build -t craftium-with-td3 .
```

Build with a specific Craftium ref:

```bash
docker build \
	--build-arg CRAFTIUM_REPO=https://github.com/mikelma/craftium.git \
	--build-arg CRAFTIUM_REF=main \
	-t craftium-with-td3 .
```

## Running Baselines

### Run the upstream TD3 benchmark code

The upstream TD3 driver can be used directly for standard Gym continuous-control tasks:

```bash
cd TD3
python main.py --env HalfCheetah-v2
```

This produces evaluation arrays in `TD3/results/` and can optionally save weights in `TD3/models/` when `--save_model` is provided.

### Run a Craftium baseline

Craftium already ships with baseline scripts for PPO-style training. For example:

```bash
cd craftium
python sb3_train.py --env-id Craftium/ChopTree-v0
```

These scripts are useful for validating that the Craftium installation works before implementing a TD3-specific adaptation.

## Suggested Project Workflow

For this repository, a reasonable development flow is:

1. Verify that Craftium builds and runs with an existing baseline script.
2. Use the upstream TD3 code to identify the assumptions it makes about observation and action spaces.
3. Create a project-specific training entry point that bridges TD3 to one Craftium environment.
4. Save outputs under [logs](./logs), [results](./results), and [analysis](./analysis) for repeatable comparisons.
5. Add project-specific tests to [tests](./tests) as the integration code matures.

## Known Gaps

At the current stage of the project:

- the root `main.py` is still only a placeholder,
- no repository-level tests have been added yet,
- no root-level experiment runner exists for TD3 on Craftium,
- `analysis/`, `benchmarks/`, `results/`, and `tests/` are mostly scaffolding directories.

That is normal for an early integration repo, but it is worth stating explicitly so that readers know where the project stands.

## References

- TD3 paper: <https://arxiv.org/abs/1802.09477>
- Craftium project: <https://github.com/mikelma/craftium>
- Craftium paper: <https://arxiv.org/abs/2407.03969>
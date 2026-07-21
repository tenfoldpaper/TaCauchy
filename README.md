# TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

**TaCauchy** is an extensible finite-element framework for vision-based tactile simulation, built on [Isaac Sim / Isaac Lab](https://isaac-sim.github.io/IsaacLab) and [UIPC](https://github.com/spiriMirror/libuipc). It extends the [TacEx](https://github.com/DH-Ng/TacEx) codebase with **direct Cauchy stress extraction**, **contact traction recovery**, **adaptive mesh refinement**, and **multi-sensor support** (GelSight Mini, DIGIT, 9DTact).

> Paper: *TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation* — accepted at **IROS 2026**.

## Highlights

- **Physics-first force fields**: Cauchy stress from hyperelastic FEM, projected to contact surfaces for normal pressure and tangential traction
- **Modular VBTS interface**: plug-in optical backends (Taxim, FOTS, ManiSkill-style FEM rendering)
- **Multi-sensor demos**: unified UIPC backend for GelSight Mini, DIGIT, and 9DTact
- **Isaac Lab integration**: RL tasks, benchmarking, and interactive shape-touch demos

## Quick Start

See **[REPRODUCTION.md](REPRODUCTION.md)** for the full installation and reproduction guide.

```bash
git clone --recurse-submodules https://github.com/figsama/TaCauchy.git
cd TaCauchy
git lfs install
./scripts/setup_assets.sh          # pull and verify Git LFS assets
./scripts/setup_libuipc_patches.sh # apply TaCauchy's local libuipc changes
conda activate env_isaaclab        # after Isaac Sim + Isaac Lab setup
./tacex.sh -i
pip install -e source/tacex_uipc -v
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight
```

## Repository Layout

| Path | Description |
|------|-------------|
| `source/tacex/` | VBTS sensor abstraction and optical simulators |
| `source/tacex_uipc/` | UIPC FEM integration, stress/contact extraction, mesh generation |
| `source/tacex_assets/` | Sensor and robot configuration plus Git LFS USD/calibration assets |
| `source/tacex_tasks/` | Isaac Lab RL environments |
| `scripts/demos/shape_touch/` | **Main TaCauchy demos** — force-field visualization and benchmarks |
| `docs/ARCHITECTURE_CN.md` | Code architecture notes (Chinese) |

## Main Demos

```bash
# Minimal demo: interpolated 20×20 force grid + tactile RGB
python scripts/demos/shape_touch/simple_tactile_demo.py --sensor gelsight

# Full modular visualization
python scripts/demos/shape_touch/contact_force_visualization_demo_modular.py

# Performance benchmark (headless)
python scripts/demos/shape_touch/benchmark_tactile_performance.py --sensor gelsight --headless
```

Sensor options: `--sensor gelsight | digit | 9dtact`

## Documentation

- [REPRODUCTION.md](REPRODUCTION.md) — installation & reproduction
- [ASSETS.md](ASSETS.md) — large binary assets policy
- [docs/source/installation/Local-Installation.md](docs/source/installation/Local-Installation.md) — detailed TacEx/UIPC install
- [MESH_REFINEMENT_PLUGIN_GUIDE.md](MESH_REFINEMENT_PLUGIN_GUIDE.md) — adaptive mesh refinement
- [TACTILE_VISUALIZATION_GUIDE.md](TACTILE_VISUALIZATION_GUIDE.md) — force-field visualization

## Citation

If you use TaCauchy, please cite our paper (IROS 2026) and acknowledge TacEx:

```bibtex
@inproceedings{tacauchy2026,
  title     = {TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation},
  author    = {},  % fill in from the published paper
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}

@article{nguyen2024tacex,
  title   = {TacEx: GelSight Tactile Simulation in Isaac Sim -- Combining Soft-Body and Visuotactile Simulators},
  author  = {Duc Huy Nguyen and Tim Schneider and Guillaume Duret and Alap Kshirsagar and Boris Belousov and Jan Peters},
  year    = {2024},
  eprint  = {2411.04776},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

## Acknowledgements

TaCauchy builds upon [TacEx](https://github.com/DH-Ng/TacEx), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [Taxim](https://github.com/Robo-Touch/Taxim), [FOTS](https://github.com/Rancho-zhao/FOTS), and [UIPC / libuipc](https://github.com/spiriMirror/libuipc).

## License

MIT License — see upstream TacEx and libuipc for component-specific licenses.

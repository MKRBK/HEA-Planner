# HEA-Planner

> **Notice:** We are writing the detailed descriptions, tutorials, simulation environments, and related hardware interfaces for our source code and other related materials, in order to foster future research in this autonomous aerial landing and robot collision avoidance line. Complete resources will be found in our repository on GitHub coming soon.

---

## HEA-Planner

**HEA-Planner** is a hierarchical environment-aware trajectory planning framework tailored for autonomous quadrotor UAV landing on moving platforms (UGVs) in confined, dynamic warehouse environments. The hierarchical framework is systematically organized into two core layers:

* **Path Planning Layer**:
  * **Global Path Planning (EA-ACO)**: Generates safe 3D reference paths balancing obstacle clearance, turning angles, and path length.
  * **Local Dynamic Replanning (EA-A\*)**: Provides spatiotemporal reactive replanning for dynamic obstacle evasion and target tracking.

* **Trajectory Optimization Layer**:
  * **Corridor Trajectory Optimization (SG-MINCO)**: Generates spatial-guided continuous trajectories that eliminate corridor corner-cutting.
  * **Moving Platform Landing (STC-MINCO)**: Optimizes spatiotemporally coupled trajectories for dynamic rendezvous and landing.

---

## Abstract

Autonomous quadrotor landing on moving unmanned ground vehicle (UGV) platforms enables air-ground logistics mission handover among heterogeneous unmanned aerial vehicle–unmanned ground vehicle (UAV–UGV) cooperative systems in confined warehouse environments. Existing trajectory planners often compromise obstacle clearance in narrow corridors or struggle to track maneuvering platforms during dynamic intrusions. To address these limitations, a hierarchical environment-aware trajectory planning framework (HEA-Planner) is presented for quadrotor landing on moving platforms. At the global layer, an Environment-Aware Ant Colony Optimization (EA-ACO) algorithm generates safe reference paths using environmental feedback and edge weighting. At the local layer, an Environment-Aware A* (EA-A*) replanner enables reactive obstacle avoidance and platform tracking via spatiotemporal risk repulsion and predictive attraction fields. At the trajectory optimization layer, a Spatially-Guided MINCO (SG-MINCO) solver preserves corridor clearance using waypoint guidance. In addition, a Spatiotemporal Coupled MINCO (STC-MINCO) solver resolves dynamic rendezvous via analytical duration total derivatives. Simulations and physical flight experiments demonstrate that the planner maintains corridor clearance, achieves an average onboard replanning time of $33.6 \pm 4.9\ \text{ms}$, and the system achieves terminal touchdown within $148.2 \pm 37.3\ \text{mm}$ on the moving platform deck.

---

## Core Algorithm Modules (`include/hea_planner`)

The library is header-only and consists of the following components:

* `ea_aco.hpp`: Global path planner (EA-ACO).
* `ea_astar.hpp`: Local dynamic replanner (EA-A*).
* `sg_minco.hpp`: Spatial-guided trajectory optimizer (SG-MINCO).
* `stc_minco.hpp`: Moving-platform landing trajectory optimizer (STC-MINCO).
* `minco_s3nu.hpp`: Polynomial trajectory generation and gradient computation.
* `lbfgs_optimizer.hpp`: L-BFGS numerical optimizer.
* `hea_config.hpp`: Parameter definitions and configuration manager.
* `common.hpp`: Basic types and obstacle models.
* `voxel_map.hpp`: 3D Euclidean distance field query interface.

---

## Prerequisites

* **C++ Standard**: C++17 or higher
* **Linear Algebra Library**: [Eigen3](https://eigen.tuxfamily.org/) ($\ge 3.3$)
* **Build System**: CMake ($\ge 3.14$)

---

## License

This project is released under the [MIT License](LICENSE).

# Three-Terminal Steiner Exact: Steiner Tree Optimization for Bounded Edge Lengths

This repository provides a Python implementation of the 3-terminal exact algorithm (AlgB) for solving the Steiner Tree Problem with Minimum Number of Steiner Points and Bounded Edge Length (STP-MSPBEL).

## Problem Background

The Steiner Tree Problem with Minimum Number of Steiner Points and Bounded Edge Length (STP-MSPBEL) requires the construction of a Steiner tree connecting $n$ terminal points in a 2D Euclidean plane. The fundamental constraint is that every edge in the resulting tree must have a length no greater than a given constant $R$. The objective is to minimize the total number of Steiner points (repeaters) added to the network to ensure connectivity.

### Theoretical Foundation

This implementation is primarily based on the research presented in:

> Shin, D., & Choi, S. (2023). *An efficient 3-approximation algorithm for the Steiner tree problem with the minimum number of Steiner points and bounded edge length.* PLOS ONE, 18(11).

The repository focuses on the **Exact algorithm for three input points (AlgB)** described in the aforementioned work. This algorithm identifies the optimal location for a Steiner junction point for three terminals in constant time by solving a specific quartic polynomial, referred to as **Equation (6)**.

## Application and Integration

The 3-terminal Steiner Exact logic implemented here serves as an optimization component within the broader **DT-PSP** (Delaunay Triangulation with Pruning, Steiner points, and Pruning) framework described by:

> Sripotchanart, R., Si, W., Calheiros, R. N., & Zhang, H. (2024). *Deploying 2-connected quantum networks with the minimum number of drone-based repeaters.*

In the context of drone-based quantum networks, this algorithm is utilized to replace Minimum Spanning Tree (MST) "wedges" with optimal "3-star" configurations. This substitution significantly reduces the total number of drone-based quantum repeaters required to maintain network connectivity under distance constraints.

## Key Components

- **Equation (6) Solver:** An analytical approach to determine the junction point ($P_j$) that minimizes the Steiner point count for three terminals.
- **Edge Steinernization:** A method for subdividing graph edges into segments of length at most $R$ by inserting the minimum required number of Steiner points.
- **Geometric Normalization:** A robust transformation engine to rotate and scale terminal triangles, ensuring consistent algebraic evaluation of the optimal junction regions.
- **Graph Integration:** Full support for `NetworkX` graph structures and `Shapely` geometric primitives for spatial analysis.

## Requirements

- `numpy`
- `shapely`
- `networkx`

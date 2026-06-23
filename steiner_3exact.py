import math
import numpy as np
import scipy.optimize as opt
from itertools import combinations
import networkx as nx
from shapely.geometry import LineString
from shapely import affinity
from typing import Tuple, Optional, Any
from geometry_utils import *

# --- 3-Exact Steiner Algorithm ---
def solve_3exact_steiner(pa: Tuple[float, float], pb: Tuple[float, float], pc: Tuple[float, float], d_max: float) -> Tuple[int, Tuple[float, float], bool]:
    """
    Find the optimal Steiner junction for three terminals under bounded-edge constraint (Equation 6).
    
    :param pa, pb, pc: Terminal points.
    :param d_max: Maximum segment length.
    :return: (repeater_count, junction_point, success_flag)
    """
    def get_ijk(a, b, c, f, r):
        return math.ceil(math.hypot(f[0] - a[0], f[1] - a[1]) / r - 1e-10), math.ceil(math.hypot(f[0] - b[0], f[1] - b[1]) / r - 1e-10), math.ceil(math.hypot(f[0] - c[0], f[1] - c[1]) / r - 1e-10)
        # return math.ceil(calc_distance(x, a) / r - 1e-10), math.ceil(calc_distance(x, b) / r - 1e-10), math.ceil(calc_distance(x, c) / r - 1e-10)
    
    pf = find_fermat_point(pa, pb, pc)
    wedge_adj = 0
    if not pf: 
        pf = pb
        wedge_adj = +1
    i_f, j_f, k_f = get_ijk(pa, pb, pc, pf, d_max)
    t_base = i_f + j_f + k_f - 2 + wedge_adj
    best_t = float('inf')
    best_pj = None

    sequences = {
                    "pa, pb, pc": (pa, pb, pc),
                    "pb, pc, pa": (pb, pc, pa),
                    "pc, pa, pb": (pc, pa, pb)
                    }

    for name, (pa, pb, pc) in sequences.items():

        # Assumed that |pbpf| >= |pcpf|, if not swap pb and pc to maintain the same logic for the intersection calculations 
        if calc_distance(pc, pf) < calc_distance(pb, pf):
                p_temp = pb
                pb = pc
                pc = p_temp


        i_f, j_f, k_f = get_ijk(pa, pb, pc, pf, d_max)
        print(f"Sequence: {name}, i_f: {i_f}, j_f: {j_f}, k_f: {k_f}, t_base: {t_base}")
        # Trial configurations from Shin & Choi 2023
        params_list = [
            (i_f-2, j_f, k_f), (i_f-3, j_f, k_f+1), (i_f-4, j_f, k_f+2), (i_f-3, j_f+1, k_f),
            (i_f-1, j_f, k_f), (i_f-2, j_f, k_f+1), (i_f-3, j_f, k_f+2), (i_f-2, j_f+1, k_f)
        ]

        for i, p in enumerate(params_list):
            t = t_base + (-2 if i < 4 else -1)
            convex_cell = None
            print(f"======== Param:", p[0], p[1], p[2], "========")
            convex_cell = _calc_eq6_feasibility(pa, pb, pc, p[0], p[1], p[2], t, d_max)
            if convex_cell:                
                p_j = circle_circle_intersection_nearest(pb, pc, convex_cell[1]*d_max, convex_cell[2]*d_max, pa)
                if p_j:
                    cost = sum(convex_cell)-2
                    nodes = [node.strip() for node in name.split(",")]
                    print(f"{nodes[1]}: ({pb[0]:.2f}, {pb[1]:.2f}), {nodes[2]}: ({pc[0]:.2f}, {pc[1]:.2f}) Ci: {convex_cell[0]}, Cj: {convex_cell[1]}, Ck: {convex_cell[2]}")                    
                    if cost < best_t:
                        print("Current Best:",best_t, "<< New Best:",cost)
                        best_t = cost
                        best_pj = p_j
                        if best_t == t and i < 4: # 3-Exact solution
                            # -2 case is the theoretical minimum (Lemma 6), so we can safely early-exit globally.
                            return best_t, best_pj, True
    if best_t != float('inf'):
        # We found at least a -1 case. Return the best found across all permutations.
        return best_t, best_pj, True

    print("No solution found.")
    return t_base, pf, False

def _calc_eq6_feasibility(pa, pb, pc, i, j, k, t, d_max, DEBUG_MODE=False):
    """
    Checks if a discrete hop count configuration is feasible by root finding.
    Safeguarded against companion matrix degradation and polynomial expansion pitfalls.
    """
    
    def solve_geometric_quartic(pa, pb, pc, d_max, j, k, t):
        # 1. Normalize points to preserve bit precision
        pa_n, pb_n, pc_n, r_n = normalize_points((pa[0], pa[1]), (pb[0], pb[1]), (pc[0], pc[1]), d_max)
        xa, ya = pa_n[0], pa_n[1]
        
        # 2. Corrected Intermediate terms
        a = (((j - k) * r_n) + 1.0) / 2.0
        b_prime = 1.0 - a - xa
        s = a * (1.0 - a) * r_n
        c0_prime = r_n * t + 1.0
        m = (2.0 * a - 1.0) * r_n
        
        a2 = m**2 + 4.0 * s * r_n - 4.0 * r_n**2
        a1 = 2.0 * m * b_prime - 4.0 * s + 4.0 * r_n * c0_prime
        a0 = b_prime**2 + ya**2 - c0_prime**2
        
        # 3. Corrected raw algebraic coefficients
        coeffs = np.array([
            a2**2, 
            2.0 * a2 * a1, 
            a1**2 + 2.0 * a2 * a0 - 16.0 * (ya**2) * s * r_n, 
            2.0 * a1 * a0 + 16.0 * (ya**2) * s, 
            a0**2
        ], dtype=np.float64)

        # --- FIX 1: Balance Coefficients (Conditioning the Companion Matrix) ---
        max_coeff = np.max(np.abs(coeffs))
        if max_coeff > 0:
            coeffs_conditioned = coeffs / max_coeff
        else:
            coeffs_conditioned = coeffs

        # --- FIX 2: Loosen Imaginary Threshold to 1e-7 ---
        raw_roots = np.roots(coeffs_conditioned)
        valid_roots = [r.real for r in raw_roots if abs(r.imag) < 1e-7]
        
        # --- FIX 3: Polish Roots via Local Refinement ---
        def unexpanded_target(x):
            term1 = a2 * (x**2) + a1 * x + a0
            # Corrected invalid square root math
            inner = s * (r_n * (x**2) - x)
            term2 = 4.0 * ya * np.sqrt(max(0.0, inner))
            return term1 - term2

        polished_roots = []
        for r in valid_roots:
            try:
                refined = opt.newton(unexpanded_target, r, tol=1e-12, maxiter=50)
                polished_roots.append(refined)
            except (RuntimeError, ZeroDivisionError):
                polished_roots.append(r)

        # roots = valid_roots 
        roots = sorted(list(set(polished_roots)))
        
        # 4. Final translation step
        sub = (j + k) / 2.0 + 1.0 / (2.0 * r_n)

        return roots, sub, xa, ya, r_n, unexpanded_target

    # Execute geometric extraction
    roots, sub, xa, ya, r_n, unexpanded_target = solve_geometric_quartic(pa, pb, pc, d_max, j, k, t)

    # Track solutions
    for idx in range(len(roots) - 1):

        if DEBUG_MODE:
            print("i,j,k,t:", i,j,k,t)
            print("Z_min:", math.ceil(roots[idx] - sub - 1e-10), "Z_max:", math.floor(roots[idx+1] - sub + 1e-10))

        mid = (roots[idx] + roots[idx+1]) / 2.0
        
        # --- FIX 4: Avoid np.polyval. Check structural feasibility on unexpanded relation ---
        if unexpanded_target(mid) <= 1e-7:
            
            # --- FIX 5: Protect range limits from epsilon micro-clipping ---
            lower_bound = math.floor(roots[idx] - sub + 1e-9)
            upper_bound = math.ceil(roots[idx+1] - sub - 1e-9)
            
            for z in range(lower_bound, upper_bound + 1):
                ci, cj, ck = i - 2*z, j + z, k + z
                
                if ci >= 0 and cj >= 0 and ck >= 0:
                    # Target centers in normalized coordinate coordinates space 
                    # Assuming baseline centers are spaced at unit distance (0,0) and (1,0)
                    p_j = circle_circle_intersection_nearest(
                        (0.0, 0.0), (1.0, 0.0), 
                        cj * r_n, ck * r_n, 
                        (xa, ya)
                    )
                    
                    if p_j:
                        dist_to_target = calc_distance(p_j, (xa, ya))
                        
                        if DEBUG_MODE: print("Pj to Pa:", dist_to_target, "Ci:", ci*r_n)
                        if dist_to_target <= (ci * r_n) + 1e-9:
                            return (ci, cj, ck)
    
    return None

def normalize_points(pa: Tuple[float, float], pb: Tuple[float, float], pc: Tuple[float, float], r: float = 1.0):
    """
    Normalize points such that pb is at the origin and pc is on the positive x-axis.
    This simplifies complex geometric calculations by placing the triangle in a
    canonical orientation.
    """
    pa, pb, pc = np.array(pa), np.array(pb), np.array(pc)
    v = pc - pb
    d = np.linalg.norm(v)
    if d == 0: raise ValueError("Points pb and pc must be distinct to establish an axis.")
    pa0, pc0 = pa - pb, pc - pb
    angle = -math.atan2(pc0[1], pc0[0])
    ca, sa = math.cos(angle), math.sin(angle)
    rot_matrix = np.array([[ca, -sa], [sa, ca]])
    pa1, pc1 = rot_matrix @ pa0, rot_matrix @ pc0
    scale = 1.0 / d
    return pa1 * scale, np.array([0.0, 0.0]), pc1 * scale, r * scale

def calc_steinernize_MST(Pa, Pb, Pc, R=1):
    """
    Calculate internal Steiner points for a Minimum Spanning Tree of three terminals.
            
    :param Pa: First terminal Point object.
    :param Pb: Second terminal Point object.
    :param Pc: Third terminal Point object.
    :param R: Maximum distance constraint (radius).
    :return: List of coordinates (x, y) for the inserted Steiner repeaters.
    """
    # Calculate pairwise distances
    d_ab = math.dist(Pa, Pb)
    d_bc = math.dist(Pb, Pc)
    d_ac = math.dist(Pa, Pc)
    
    # Define candidate segments with their lengths
    candidates = [
        ((Pa, Pb), d_ab),
        ((Pb, Pc), d_bc),
        ((Pa, Pc), d_ac)
    ]
    
    # Sort candidates by length in ascending order
    candidates.sort(key=lambda x: x[1])
    
    # Select the two shorter segments
    edges = [candidates[0][0], candidates[1][0]]
    
    extra_steiners = []
    
    for u, v in edges:
        
        x1, y1 = u[0], u[1]
        x2, y2 = v[0], v[1]
        dx, dy = (x2 - x1), (y2 - y1)

        # Prefer stored length; fall back to Euclidean if missing
        L = math.hypot(dx, dy)

        if round(L, 9) <= R:
            # Already within span; no extra steiners needed
            continue

        # Number of interior points so that each gap <= lmax
        # This formula works uniformly for all L:
        # if lmax < L <= 2*lmax -> n = 1 (midpoint),
        # if 2*lmax < L <= 3*lmax -> n = 2, etc.
        n = math.ceil(round(L / R, 9)) - 1

        # Parametric positions along the segment at equal spacing
        step = 1.0 / (n + 1)
        for i in range(1, n + 1):
            t = i * step
            extra_steiners.append((x1 + dx * t, y1 + dy * t))
            
    return extra_steiners

def calc_steinernize_3ST(Pa, Pb, Pc, Px, R=1):
    """
    Calculate internal Steiner points for a 3-star network.
            
    :param Pa: First terminal Point object.
    :param Pb: Second terminal Point object.
    :param Pc: Third terminal Point object.
    :param Px: Center Steiner Point object.
    :param R: Maximum distance constraint (radius).
    :return: A tuple containing the total number of repeaters and a list of coordinates 
             (x, y) for the inserted Steiner repeaters.
    """
    edges = [(Pa, Px), (Pb, Px), (Pc, Px)]
    extra_steiners = []
    
    for u, v in edges:
        x1, y1 = u[0], u[1]
        x2, y2 = v[0], v[1]
        dx, dy = (x2 - x1), (y2 - y1)
        L = math.hypot(dx, dy)

        if round(L, 9) <= R:
            continue

        n = math.ceil(round(L / R, 9)) - 1
        step = 1.0 / (n + 1)
        for i in range(1, n + 1):
            t = i * step
            extra_steiners.append((x1 + dx * t, y1 + dy * t))
            
    total_steiners = len(extra_steiners) + 1 # +1 for the center Steiner point Px
    return total_steiners, extra_steiners
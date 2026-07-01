import math
import numpy as np
from shapely.geometry import LineString
from shapely import affinity
from typing import Tuple, Optional, Any

def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Calculate the angle at p2 formed by p1, p2, and p3 in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.dist(p1, p2)
    mag_v2 = math.dist(p3, p2)
    if mag_v1 == 0 or mag_v2 == 0: return 0.0
    cos_theta = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
    return math.degrees(math.acos(cos_theta))

def find_equilateral_triangle_vertex(a: Tuple[float, float], b: Tuple[float, float], cw: int = 1) -> Tuple[float, float]:
    """Find the third vertex of an equilateral triangle given two vertices."""
    base_vector = LineString([a, b])
    rotated_vector = affinity.rotate(base_vector, 60 * cw, origin=(a[0], a[1]))
    return rotated_vector.coords[1]

def find_fermat_point(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Find the Fermat point (Torricelli point) of a triangle."""
    def round_point(p, digits=9): return (round(p[0], digits), round(p[1], digits))
    for cw in [1, -1]:
        p4 = find_equilateral_triangle_vertex(p1, p2, cw)
        p5 = find_equilateral_triangle_vertex(p2, p3, cw)
        p6 = find_equilateral_triangle_vertex(p3, p1, cw)
        l1, l2, l3 = LineString([p3, p4]), LineString([p1, p5]), LineString([p2, p6])
        i1, i2, i3 = l1.intersection(l2), l1.intersection(l3), l2.intersection(l3)
        if not (i1.is_empty or i2.is_empty or i3.is_empty):
            i1, i2, i3 = round_point(i1.coords[0]), round_point(i2.coords[0]), round_point(i3.coords[0])
            if math.dist(i1, i2) < 1e-7 and math.dist(i1, i3) < 1e-7: return i1
    return None

def circle_circle_intersection_nearest_old(c1: Tuple[float, float], c2: Tuple[float, float], r1: float, r2: float, ref_p: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Find the intersection point of two circles nearest to a reference point."""
    p1, p2, p_ref = np.array((c1[0], c1[1])), np.array((c2[0], c2[1])), np.array((ref_p[0], ref_p[1]))
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0: return None
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d
    i1, i2 = p_mid + h * perp, p_mid - h * perp
    return tuple(i1) if np.linalg.norm(i1 - p_ref) < np.linalg.norm(i2 - p_ref) else tuple(i2)

def circle_circle_intersection_nearest(
    c1: Tuple[float, float], c2: Tuple[float, float], r1: float, r2: float, ref_p: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """
    Find the intersection point of two circles nearest to a reference point 
    with safeguards against catastrophic floating-point cancellation.
    """
    p1 = np.array((c1[0], c1[1]), dtype=np.float64)
    p2 = np.array((c2[0], c2[1]), dtype=np.float64)
    p_ref = np.array((ref_p[0], ref_p[1]), dtype=np.float64)
    d = math.dist(p1, p2)
    
    # Tolerance checking for floating point boundaries
    EPS = 1e-9
    if d > (r1 + r2) + EPS or d < abs(r1 - r2) - EPS or d < EPS:
        # print(f"d:{d} , (r1+r2):{r1+r2} , abs(r1-r2):{abs(r1-r2)} , EPS:{EPS}")
        return None
        
    # High-precision alternative to: (r1**2 - r2**2 + d**2)
    # Factoring difference of squares completely prevents bit truncation
    diff_squares = (r1 - r2) * (r1 + r2)
    a = (diff_squares + d**2) / (2.0 * d)
    
    # High-precision calculation of perpendicular height h
    # Factoring r1**2 - a**2 to (r1 - a)*(r1 + a) stabilizes near-tangent states
    h_squared = (r1 - a) * (r1 + a)
    h = math.sqrt(max(0.0, h_squared))
    
    # Vector arithmetic
    p_mid = p1 + a * (p2 - p1) / d
    perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]], dtype=np.float64) / d
    
    i1 = p_mid + h * perp
    i2 = p_mid - h * perp
    
    # Select closest to reference and unpack explicitly to avoid Shapely type-casting drift
    if np.linalg.norm(i1 - p_ref) < np.linalg.norm(i2 - p_ref):
        return (float(i1[0]), float(i1[1]))
    else:
        return (float(i2[0]), float(i2[1]))

def calc_distance(p1, p2):
    """
    Calculate the Euclidean distance between two coordinate tuples.
    """
    return math.dist(p1, p2)
def triangle_angles(A, B, C):
    """
    Calculate all angles of a triangle formed by points A, B, and C.
    """
    angle_A = calculate_angle(B, A, C)
    angle_B = calculate_angle(A, B, C)
    angle_C = calculate_angle(A, C, B)
    return angle_A, angle_B, angle_C

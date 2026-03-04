import numpy as np
from typing import Iterable, Tuple, Union

Point = Tuple[float, float]


def homography_from_points(src_pts: Union[Iterable[Point], np.ndarray],
                           dst_pts: Union[Iterable[Point], np.ndarray]) -> np.ndarray:
    """
    Estima a homografia H (3x3) que mapeia pontos do plano src -> dst, usando N >= 4 correspondências.
    Resolve por mínimos quadrados (Least Squares).

    src_pts, dst_pts: iteráveis (ou np.ndarray) com shape (N,2), N>=4
    Retorna: H (3x3) com H[2,2] = 1
    """
    src = np.asarray(list(src_pts), dtype=np.float64)
    dst = np.asarray(list(dst_pts), dtype=np.float64)

    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 2 or dst.shape[1] != 2:
        raise ValueError("src_pts e dst_pts devem ter shape (N, 2).")
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src_pts e dst_pts devem ter o mesmo número de pontos.")
    if src.shape[0] < 4:
        raise ValueError("São necessários pelo menos 4 pontos.")

    N = src.shape[0]

    A = np.zeros((2 * N, 8), dtype=np.float64)
    b = np.zeros((2 * N,), dtype=np.float64)

    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        r = 2 * i
        # u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
        A[r, :] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        b[r] = u

        # v = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)
        A[r + 1, :] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[r + 1] = v

    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8:
        raise ValueError("Não foi possível estimar H (pontos degenerados, ex: muitos colineares).")

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ], dtype=np.float64)

    return H


def apply_homography_to_point(pt: Point, H: np.ndarray) -> Point:
    """
    Aplica H em um ponto (x,y). Retorna (u,v).
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("H deve ter shape (3,3).")

    x, y = float(pt[0]), float(pt[1])
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p

    if abs(q[2]) < 1e-12:
        raise ValueError("w ~ 0 ao aplicar homografia (ponto no infinito/instável).")

    return (float(q[0] / q[2]), float(q[1] / q[2]))


def apply_homography_to_points(pts: Union[Iterable[Point], np.ndarray], H: np.ndarray) -> np.ndarray:
    """
    Aplica H em vários pontos. Retorna np.ndarray shape (M,2).
    """
    H = np.asarray(H, dtype=np.float64)
    P = np.asarray(list(pts), dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("pts deve ter shape (M,2).")

    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    Ph = np.hstack([P, ones])                # (M,3)
    Qh = (H @ Ph.T).T                        # (M,3)

    w = Qh[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("Algum ponto mapeou com w ~ 0 (instável).")

    Q = Qh[:, :2] / w
    return Q


# -------------------------
# Exemplo rápido
# -------------------------
if __name__ == "__main__":
    # N pontos (>=4)
    src_pts = [(100, 200), (500, 220), (520, 700), (90, 680), (300, 450)]
    dst_pts = [(0, 0), (800, 0), (800, 500), (0, 500), (400, 250)]

    H = homography_from_points(src_pts, dst_pts)

    p = (350, 400)
    print("H=\n", H)
    print("p ->", apply_homography_to_point(p, H))

    pts = [(120, 210), (480, 650), (250, 500)]
    print("pts ->\n", apply_homography_to_points(pts, H))
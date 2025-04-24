"""
step1_project_edges.py
------------------------------------------------------------
0.  Grab a frame (camera or static image).
1.  Canny edge map.
2.  Probabilistic-Hough segments.
3.  Draw segments on a black canvas → "line image".
4.  Warp line image into projector space and display it
    full-screen, while also showing live previews:
        • Camera   (raw)
        • Edges    (Canny)
        • Lines    (segments overlay)
Esc or q = quit.
"""

from __future__ import annotations
import platform, sys
import cv2 as cv
import numpy as np
from itertools import combinations


# ----------------------------- CONFIG -----------------------------
USE_STATIC_IMAGE   = False
STATIC_IMAGE_PATH  = "img1.png"
CAMERA_INDEX       = 1
H_CP_PATH          = "H_cp.npy"          # 3×3 camera→projector homography

PROJ_RES      = (1280, 720)              # (width, height)
PROJ_X_OFFSET = 1920                     # projector desktop origin (x,y)
PROJ_Y_OFFSET = 0

# Edge / Hough parameters (tweak as needed)
CANNY_LOW, CANNY_HIGH  = 40, 50
HOUGH_THRESH           = 10
HOUGH_MIN_LEN, HOUGH_GAP = 5, 10
# ------------------------------------------------------------------

def get_frame() -> np.ndarray:
    """Return a single BGR frame."""
    if USE_STATIC_IMAGE:
        img = cv.imread(STATIC_IMAGE_PATH, cv.IMREAD_COLOR)
        if img is None:
            sys.exit(f"Cannot read {STATIC_IMAGE_PATH}")
        return img

    api = cv.CAP_DSHOW if platform.system() == "Windows" else 0
    cap = cv.VideoCapture(CAMERA_INDEX, api)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera #{CAMERA_INDEX}")
    for _ in range(5): cap.read()                      # warm-up
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Camera read failed")
    return frame

def filter_parallel_close(lines, angle_thresh_deg: float = 10.0):
    """
    Remove the shorter of any two line segments that are
    (i)   parallel within `angle_thresh_deg`,  and
    (ii)  closer than 1/5 × L_max,   where L_max is the longest
          segment in the entire set.
    """
    L = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    if L.shape[1] != 4:
        raise ValueError("lines must be (...,4) array‐like")

    # --- precompute
    dx, dy   = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    length   = np.hypot(dx, dy)
    max_len  = length.max()
    dist_thr = 0.2 * max_len                 # 1/5 × L_max
    angle    = (np.arctan2(dy, dx) + np.pi) % np.pi  # wrap to [0,π)
    ang_eps  = np.deg2rad(angle_thresh_deg)

    keep = np.ones(len(L), bool)

    def pt_line_dist(px, py, x1, y1, x2, y2):
        return abs((y2 - y1) * px - (x2 - x1) * py + x2*y1 - y2*x1) \
               / np.hypot(x2 - x1, y2 - y1)

    for i in range(len(L)):
        if not keep[i]:
            continue
        for j in range(i+1, len(L)):
            if not keep[j]:
                continue
            # parallel?
            if abs(angle[i] - angle[j]) > ang_eps:
                continue
            # ensure we measure distance from shorter to longer
            if length[i] >= length[j]:
                long_idx, short_idx = i, j
            else:
                long_idx, short_idx = j, i
            xl1, yl1, xl2, yl2 = L[long_idx]
            xs1, ys1, xs2, ys2 = L[short_idx]
            d1 = pt_line_dist(xs1, ys1, xl1, yl1, xl2, yl2)
            d2 = pt_line_dist(xs2, ys2, xl1, yl1, xl2, yl2)
            if min(d1, d2) < dist_thr:
                keep[short_idx] = False
    return L[keep]

def cluster_endpoints(lines: np.ndarray, fraction: float = 0.25) -> list[tuple[float, float]]:

    L = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    if L.shape[1] != 4:
        raise ValueError("lines must be (N,4)")

    # 1. collect endpoints
    p1 = L[:, 0:2]
    p2 = L[:, 2:4]
    pts = np.vstack((p1, p2))               # (2N,2)

    # 2. distance threshold
    seg_len = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    thresh  = fraction * seg_len.max()

    # 3. agglomerative clustering (single-link)
    n = len(pts)
    clusters: list[list[int]] = []
    unassigned = list(range(n))

    while unassigned:
        seed = unassigned.pop()             # start a new cluster
        stack = [seed]
        cluster = [seed]

        while stack:
            idx = stack.pop()
            # squared Euclidean distance to remaining points
            d2 = np.sum((pts[unassigned] - pts[idx])**2, axis=1)
            nearby_mask = d2 < thresh**2
            nearby_idx  = [unassigned[i] for i, flag in enumerate(nearby_mask) if flag]
            for k in nearby_idx:
                unassigned.remove(k)
                stack.append(k)
                cluster.append(k)

        clusters.append(cluster)

    # 4. centroids
    centroids = [tuple(pts[c].mean(axis=0)) for c in clusters]
    return centroids

def order_points_clockwise(pts):
    """
    Reorders an array of 4 points (shape (4,2)) into clockwise order.
    """
    # Compute the centroid
    center = np.mean(pts, axis=0)
    # Sort based on angle relative to the centroid.
    # Using reverse=True to have a clockwise order.
    sorted_pts = sorted(pts, key=lambda pt: np.arctan2(pt[1] - center[1], pt[0] - center[0]), reverse=True)
    return np.array(sorted_pts, dtype=np.float32)

def is_opposite_edges_parallel(ordered_quad, threshold_deg=10):
    """
    Checks if opposite edges of a 4‑point polygon (in CW order) are parallel or anti‑parallel
    within the given angular threshold.
    """
    # Build normalized edge vectors
    edges = []
    for i in range(4):
        v = ordered_quad[(i+1)%4] - ordered_quad[i]
        norm = np.linalg.norm(v) + 1e-6
        edges.append(v / norm)
    
    # Opposite edges are 0↔2 and 1↔3
    dot02 = abs(np.dot(edges[0], edges[2]))
    dot13 = abs(np.dot(edges[1], edges[3]))
    
    # Convert to deviation angle from perfect parallelism
    ang02 = np.degrees(np.arccos(np.clip(dot02, -1.0, 1.0)))
    ang13 = np.degrees(np.arccos(np.clip(dot13, -1.0, 1.0)))
    
    return (ang02 < threshold_deg) and (ang13 < threshold_deg)

def create_quadrilaterals(points, parallel_threshold=15):
    valid_quads = []
    
    if len(points) < 5:
        max_candidates = 1
    elif len(points) > 6:
        max_candidates = 3
    else:
        max_candidates = 2

    # 1) Generate all 4‑point combinations
    for comb in combinations(points, 4):
        quad = np.array(comb, dtype=np.float32)
        ordered_quad = order_points_clockwise(quad)
        
        # 2) Convexity check
        if not cv.isContourConvex(ordered_quad):
            continue
        
        # 3) Parallel‑edge check
        if not is_opposite_edges_parallel(ordered_quad, threshold_deg=parallel_threshold):
            continue
        
        # 4) Compute area and store
        area = cv.contourArea(ordered_quad)
        valid_quads.append((area, ordered_quad))
    
    # 5) Sort by area, ascending, and keep only the smallest max_candidates
    valid_quads.sort(key=lambda x: x[0])
    selected = valid_quads[:max_candidates]
    
    # 6) Extract just the quad arrays
    quad_list = [quad for (_, quad) in selected]
    
    return quad_list, len(quad_list)


def compute_homography_from_image(square_img, quad_points):
    """
    Computes a 3×3 homography mapping the corners of `square_img` onto `quad_points`.

    Parameters:
        square_img   : numpy.ndarray   your source image (square or rectangle)
        quad_points  : array‑like of shape (4,2)  the destination quadrilateral, in the same
                       clockwise order as src_pts below

    Returns:
        H : 3×3 homography matrix
    """
    h, w = square_img.shape[:2]

    # define the four corners of the source image in pixel coords
    src_pts = np.array([
        [0,   0   ],   # top‑left
        [w-1, 0   ],   # top‑right
        [w-1, h-1 ],   # bottom‑right
        [0,   h-1 ]    # bottom‑left
    ], dtype=np.float32)

    # make sure your quad_points are also in the same (clockwise) order:
    dst_pts = np.array(quad_points, dtype=np.float32)

    # compute the homography
    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    return H


def apply_homography(H, workspace, square_img):
    """
    Warps the square_img using homography H into a new image of workspace size.
    The non-transformed background remains black.
    """
    ws_width, ws_height = workspace
    warped = np.zeros((ws_height, ws_width, 3), dtype=np.uint8)
    transformed = cv.warpPerspective(square_img, H, (ws_width, ws_height),
                                       borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv.cvtColor(transformed, cv.COLOR_BGR2GRAY) > 0
    warped[mask] = transformed[mask]
    return warped

def combine_images(image_list):
    """
    Combines multiple images (of the same size) by summing their non-black pixels.
    """
    if len(image_list) == 0:
        return None
    combined = np.zeros_like(image_list[0], dtype=np.float32)
    for img in image_list:
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = gray > 0
        for c in range(3):
            combined[:, :, c][mask] += img[:, :, c][mask]
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    return combined

#--og
def detect_edges_and_lines(bgr: np.ndarray):
    """Return (edges_gray, segments, line_canvas_BGR)."""
    gray  = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (15,15),0)

    edges = cv.Canny(gray, CANNY_LOW, CANNY_HIGH)

    segs = cv.HoughLinesP(edges, 1, np.pi/180,
                          threshold     = HOUGH_THRESH,
                          minLineLength = HOUGH_MIN_LEN,
                          maxLineGap    = HOUGH_GAP)
    
    # print("segs")
    # print(segs)
    # print("segs-over")
#----------------------------------------------------
    x_low = 100
    x_high = 500
    y_low = 100
    y_high = 400

    segs_copy = []
    
    for seg in segs[:,0]:
        if not (seg[0] < x_low or seg[0] > x_high or seg[1] < y_low or seg[1] > y_high or seg[2] < x_low or seg[2] > x_high or seg[3] < y_low or seg[3] > y_high):
            segs_copy += [[seg]]
    segs = np.array(segs_copy)
    
#------------------------------------------------------
    # Draw segments on black canvas (white lines)
    canvas = np.zeros_like(bgr)
    if segs is not None:
        for x1, y1, x2, y2 in segs[:, 0]:
            cv.line(canvas, (x1, y1), (x2, y2),
                    color=(255, 255, 255), thickness=2)
    return edges, segs, canvas

def warp_to_projector(img: np.ndarray, H_cp: np.ndarray) -> np.ndarray:
    """Homography warp → projector raster size."""
    return cv.warpPerspective(img, H_cp, PROJ_RES,
                              flags=cv.INTER_LINEAR,
                              borderMode=cv.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

def main() -> None:
    # 1. load homography
    try:
        H_cp = np.load(H_CP_PATH).astype(np.float32)
        if H_cp.shape != (3, 3):
            raise ValueError
    except Exception:
        sys.exit("H_cp.npy missing or not a 3×3 matrix.")

    # 2. create windows
    cv.namedWindow("Camera", cv.WINDOW_NORMAL)
    cv.resizeWindow("Camera", 640, 360)

    cv.namedWindow("Edges",  cv.WINDOW_NORMAL)
    cv.resizeWindow("Edges", 640, 360)

    cv.namedWindow("Lines",  cv.WINDOW_NORMAL)
    cv.resizeWindow("Lines", 640, 360)

    cv.namedWindow("Projector", cv.WINDOW_NORMAL)
    cv.moveWindow("Projector", PROJ_X_OFFSET, PROJ_Y_OFFSET)
    cv.resizeWindow("Projector", *PROJ_RES)
    cv.setWindowProperty("Projector",
                         cv.WND_PROP_FULLSCREEN,
                         cv.WINDOW_FULLSCREEN)

    # 3. main loop
    while True:
        frame = get_frame()

        edges, segs, line_img = detect_edges_and_lines(frame)

        lkeep = filter_parallel_close(segs)

        centroids = cluster_endpoints(lkeep)

        canvas = np.zeros_like(frame)
        for pt in centroids:
            cv.circle(canvas, (int(pt[0]), int(pt[1])), 1, color=(255, 255, 255), thickness=2)

        quads, lenq = create_quadrilaterals(centroids)

        # pts = clean_points(extract_points_from_lines(segs))  # Clean extracted points
        print(quads)
        print(lenq)
            
        canvas1 = np.zeros_like(frame)
        if isinstance(quads, tuple):
            quads = quads[0]

        for quad in quads:
            pts = quad.reshape(-1, 1, 2).astype(np.int32)
            cv.polylines(canvas1, [pts], isClosed=True, color=(255, 255, 255), thickness=2)


        proj = warp_to_projector(canvas1, H_cp)

        # previews
        cv.imshow("Camera", frame)
        # cv.imshow("Edges",  edges)
        cv.imshow("Lines",  canvas)
        cv.imshow("Points",  canvas1)
        # cv.imshow("Lines",  line_img)
        # projector output
        cv.imshow("Projector", proj)

        if cv.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

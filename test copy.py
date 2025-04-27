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
import shapely
import shapely.ops
from sklearn.cluster import DBSCAN

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
HOUGH_MIN_LEN, HOUGH_GAP = 15, 5
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


def cluster_endpoints(lines: np.ndarray,
                      fraction: float = 0.25) -> list[tuple[float, float]]:

    L = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    if L.shape[1] != 4:
        raise ValueError("lines must be (N,4)")

    dist_arr = [    np.linalg.norm((x2 - x1, y2 - y1))    for x1, y1, x2, y2 in L]
    med_dist = np.median(dist_arr)

    eps_val = 0.33

    # 1. collect endpoints
    p1 = L[:, 0:2]
    p2 = L[:, 2:4]
    pts = np.vstack((p1, p2))  # (2N,2)
    # print(pts)
    clusters = DBSCAN(eps=eps_val * med_dist, min_samples=1).fit(pts)
    x_sums = []
    y_sums = []
    counts = []
    for i in range(len(clusters.labels_)):
        if clusters.labels_[i] >= len(x_sums):
            x_sums += [pts[i][0]]
            y_sums += [pts[i][1]]
            counts += [1]
        else:
            x_sums[clusters.labels_[i]] += pts[i][0]
            y_sums[clusters.labels_[i]] += pts[i][1]
            counts[clusters.labels_[i]] += 1

    for i in range(len(x_sums)):
        x_sums[i] /= counts[i]
        y_sums[i] /= counts[i]

    for i in range(len(L)):
        L[i][0] = x_sums[clusters.labels_[i]]
        L[i][1] = y_sums[clusters.labels_[i]]
        L[i][2] = x_sums[clusters.labels_[i + len(L)]]
        L[i][3] = y_sums[clusters.labels_[i + len(L)]]

    return L


def split_lines_at_nearby_points(raw_lines):

    L = np.asarray(raw_lines, dtype=np.float64).reshape(-1, 4)
    if L.shape[1] != 4:
        raise ValueError("lines must be (N,4)")

    dist_arr = [    np.linalg.norm((x2 - x1, y2 - y1))    for x1, y1, x2, y2 in L]
    med_dist = np.median(dist_arr)
    eps_val = 0.33

    threshold = med_dist * eps_val

    # 1. Build LineStrings and collect all endpoints
    lines = [shapely.LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in raw_lines]
    endpoints = [shapely.Point(ls.coords[0]) for ls in lines] + [shapely.Point(ls.coords[-1]) for ls in lines]

    new_segments = []
    for ls in lines:
        start_pt = shapely.Point(ls.coords[0])
        end_pt   = shapely.Point(ls.coords[-1])
        did_split = False

        for pt in endpoints:
            # skip its own endpoints
            if pt.equals(start_pt) or pt.equals(end_pt):
                continue

            # if 'pt' lies near the interior of 'ls'
            if ls.distance(pt) <= threshold:
                # manually form two new lines:
                #   segment A: start → pt
                new_segments.append((start_pt.x, start_pt.y, pt.x, pt.y))
                #   segment B: pt → end
                new_segments.append((pt.x, pt.y, end_pt.x, end_pt.y))
                did_split = True
                break

        # if no split happened, preserve the original line
        if not did_split:
            new_segments.append((start_pt.x, start_pt.y, end_pt.x, end_pt.y))

    return new_segments

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

def create_quads(lines):
    quads = []
    combos = list(combinations(lines, 4))
    for test_quad in combos:
        unique_points = set()
        for line in test_quad:
            x1, y1, x2, y2 = line
            unique_points.add((x1, y1))
            unique_points.add((x2, y2))
        if len(unique_points) == 4:
            # quads += [point for point in unique_points]
            quads.append(list(unique_points))
            # quads = [[(211, 112), ( 102, 345), (543, 765), (143, 423)]]
    return quads


def compute_homography_from_image(square_img, quad_points):
    """
    Computes a 3×3 homography mapping the corners of ⁠ square_img ⁠ onto ⁠ quad_points ⁠.

    Parameters:
        square_img   : numpy.ndarray   your source image (square or rectangle)
        quad_points  : array‑like of shape (4,2)  the destination quadrilateral, in the same
                       clockwise order as src_pts below

    Returns:
        H : 3×3 homography matrix
    """
    # print(f"quadlines{quad_lines}")
    '''
    quad_points = []
    for line in quad_lines:
        if 
        x1, y1, x2, y2 = line
        quad_points += [[x1, y1]]
    
    # print(f"quadpoints{quad_points}")
    '''

    # print(quad_points)
        
    quad_points = order_points_clockwise(quad_points)
    # print(quad_points)
        
    h, w = square_img.shape[:2]

    # define the four corners of the source image in pixel coords
    src_pts = np.array([
        [0,   0   ],   # top‑left
        [w-1, 0   ],   # top‑right
        [w-1, h-1 ],   # bottom‑right
        [0,   h-1 ]    # bottom‑left
    ], dtype=np.float32)

    # src_pts = np.array([
    #     [w-1, 0   ],   # top‑right
    #     [0,   0   ],   # top‑left
    #     [0,   h-1 ],   # bottom‑left
    #     [w-1, h-1 ]    # bottom‑right
    # ], dtype=np.float32)

    # make sure your quad_points are also in the same (clockwise) order:
    dst_pts = np.array(quad_points, dtype=np.float32)
    # print(f"sourcepts{src_pts}")
    # print(f"distpts{dst_pts}")

    # compute the homography
    H = cv.getPerspectiveTransform(src_pts, dst_pts)
    return H

def apply_homography(H, workspace, square_img):
    """
    Warps the square_img using homography H into a new image of workspace size.
    The non-transformed background remains black.
    """
    ws_width, ws_height = workspace
    
    if square_img.shape[2] == 4:
        square_rgb = square_img[:, :, :3]
        square_alpha = square_img[:, :, 3]  # extract alpha channel
    else:
        square_rgb = square_img
        square_alpha = np.ones(square_img.shape[:2], dtype=np.uint8) * 255

    # warped = np.zeros((ws_height, ws_width, 3), dtype=np.uint8)
    
    warped_rgb = cv.warpPerspective(square_rgb, H, (ws_width, ws_height),
                                    borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv.warpPerspective(square_alpha, H, (ws_width, ws_height),
                                       borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))

    warped = np.zeros((ws_height, ws_width, 3), dtype=np.uint8)
    mask = warped_alpha > 0

    for c in range(3):
        warped[:, :, c][mask] = warped_rgb[:, :, c][mask]
    
    # transformed = cv.warpPerspective(square_img, H, (ws_width, ws_height),
    #                                    borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # mask = cv.cvtColor(transformed, cv.COLOR_BGR2GRAY) > 0
    # warped[mask] = transformed[mask]
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

    edges = cv.Canny(gray, CANNY_LOW, CANNY_HIGH, L2gradient=True)

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
    # cv.namedWindow("Camera", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Camera", 640, 360)

    cv.namedWindow("Edges", cv.WINDOW_NORMAL)
    cv.resizeWindow("Edges", 640, 360)

    # cv.namedWindow("Lines",  cv.WINDOW_NORMAL)
    # cv.resizeWindow("Lines", 640, 360)

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

        lines_1 = cluster_endpoints(lkeep)

        lines_final = split_lines_at_nearby_points(lines_1)

        quads = create_quads(lines_final)

        print(f"len_quads: {len(quads)}")

        image_list = []
        for i, q in enumerate(quads):
            # print(i)
            if i > 2:
                break
            image = cv.imread(f"square{i+1}.png", cv.IMREAD_UNCHANGED)
            image_list.append(image)

        H_list = []
        for i,image in enumerate(image_list):
            # print(image_list[i])
            H = compute_homography_from_image(image_list[i], quads[i]).astype(np.float32)
            H_list.append(H)
            # print(f'H_list{i}: {H}')

        warped_list = []
        for i, H in enumerate(H_list):
            warped = apply_homography(H, frame.shape[1::-1], image_list[i])
            warped_list.append(warped)

        montage = combine_images(warped_list)
        if montage is None:
            continue
        
        # print(f"quads {quads}")
        canvas = np.zeros_like(frame)

        # for i, q in enumerate(quads):
        #     color = (0,255,255)
        #     if i == 0:
        #         color = (255,255,0)
        #     elif i == 1:
        #         color = (255,0,255)
        #     for l in q:
        #         cv.line(canvas1, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])),
        #                 color=color, thickness=1)

        proj = warp_to_projector(montage, H_cp)


        #_______PROJ FGURING OUT

        brightness_factor = 0.5  # or 0.1, adjust as needed
        proj = np.clip(proj * brightness_factor, 0, 255).astype(np.uint8)

        
        # print(proj)
        # previews
        # cv.imshow("Camera", frame)
        cv.imshow("Edges",  proj)
        cv.imshow("Lines",  line_img)
        # cv.imshow("Points",  canvas1)
        # cv.imshow("Lines",  line_img)
        # projector output
        # cv.imshow("Edges", proj)
        if len(quads) < 7:
            cv.imshow("Projector", proj)
        else:
            cv.imshow("Projector", canvas)

        if cv.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

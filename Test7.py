import cv2
import numpy as np
from itertools import combinations

debug = True

# ---------------------- CONFIG ----------------------------------------
USE_STATIC_IMAGE = True           # True → load STATIC_IMAGE_PATH once
STATIC_IMAGE_PATH = "img1.png"
CAMERA_INDEX      = 1              # ignored when USE_STATIC_IMAGE = True

TEXTURE_PATH      = "square_image.jpg"

H_CP_PATH         = "H_cp.npy"     # camera→projector homography
PROJ_SIZE         = (1280, 720)    # projector resolution  (w, h)

# where the projector’s desktop starts (pixels, relative to primary)
PROJ_X_OFFSET     = 1920           # example: primary width = 1920
PROJ_Y_OFFSET     = 0
# ----------------------------------------------------------------------

# ----------------------- Low-level helpers ----------------------------
def process_edges(image, low_threshold=50, high_threshold=200):
    """
    Takes an image and returns an edge image using the Canny edge detector.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    edged = cv2.Canny(gray, low_threshold, high_threshold)
    return edged

def extract_points_from_lines(edge_image, min_line_length=50, max_line_gap=10):
    """
    Detects line segments using probabilistic Hough transform and extracts endpoints.
    """
    lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180, threshold=80,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))
    points = np.array(points)
    unique_points = []
    tol = 5  # tolerance in pixels
    for pt in points:
        if not any(np.linalg.norm(np.array(pt) - np.array(up)) < tol for up in unique_points):
            unique_points.append(tuple(pt))
    return unique_points

def clean_points(points, desired_range=(4, 7)):
    """
    Reduces a list of points to a set within the desired range using k-means clustering.
    """
    num_points = len(points)
    min_pts, max_pts = desired_range
    target = min(max(min_pts, num_points // 2), max_pts)
    
    points_np = np.float32(points)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if num_points > target:
        ret, labels, centers = cv2.kmeans(points_np, target, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        cleaned_points = [tuple(center) for center in centers]
    else:
        cleaned_points = points
    return cleaned_points

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

def is_opposite_edges_parallel(ordered_quad, threshold_deg=15):
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


def create_quadrilaterals(points, parallel_threshold=15, max_candidates=3):
    """
    From 4-7 points, test all 4-point combinations and return only those that form
    convex quadrilaterals with opposite edges parallel (or anti‑parallel) within a threshold.
    Finally, select the `max_candidates` quads with the smallest area.
    
    Parameters:
        points (list of tuple): List of (x, y) points.
        parallel_threshold (float): Max angle (degrees) between opposite edges.
        max_candidates (int): How many smallest-area quads to return.
        
    Returns:
        quad_list (list of np.ndarray): Up to `max_candidates` 4×2 quads in CW order.
        count (int): Number of quads returned.
    """
    valid_quads = []
    
    # 1) Generate all 4‑point combinations
    for comb in combinations(points, 4):
        quad = np.array(comb, dtype=np.float32)
        ordered_quad = order_points_clockwise(quad)
        
        # 2) Convexity check
        if not cv2.isContourConvex(ordered_quad):
            continue
        
        # 3) Parallel‑edge check
        if not is_opposite_edges_parallel(ordered_quad, threshold_deg=parallel_threshold):
            continue
        
        # 4) Compute area and store
        area = cv2.contourArea(ordered_quad)
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
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H


def apply_homography(H, workspace, square_img):
    """
    Warps the square_img using homography H into a new image of workspace size.
    The non-transformed background remains black.
    """
    ws_width, ws_height = workspace
    warped = np.zeros((ws_height, ws_width, 3), dtype=np.uint8)
    transformed = cv2.warpPerspective(square_img, H, (ws_width, ws_height),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY) > 0
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        for c in range(3):
            combined[:, :, c][mask] += img[:, :, c][mask]
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    return combined


#------------------------------------------------------MAIN
def main():
    # Load resources ---------------------------------------------------
    H_cp = np.load(H_CP_PATH)  # Load the camera-to-projector homography matrix
    texture = cv2.imread(TEXTURE_PATH)  # Read the texture image
    if texture is None:
        raise IOError(f"Cannot load texture '{TEXTURE_PATH}'")

    # Projector output window -----------------------------------------
    cv2.namedWindow("ProjectorOut", cv2.WINDOW_NORMAL)
    cv2.moveWindow("ProjectorOut", PROJ_X_OFFSET, PROJ_Y_OFFSET)
    cv2.setWindowProperty("ProjectorOut", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Choose input source (static image or live camera feed) -----------
    if USE_STATIC_IMAGE:
        frame = cv2.imread(STATIC_IMAGE_PATH)
        if frame is None:
            raise IOError(f"Cannot load '{STATIC_IMAGE_PATH}'")
        frames = [frame]  # single iteration list for static image
        cap = None
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise IOError(f"Camera #{CAMERA_INDEX} not accessible")
        frames = iter(lambda: cap.read()[1], None)

    try:
        for frame in frames:
            if frame is None:
                break

            # Process the edges and extract points from the current frame
            edges = process_edges(frame)  # Process the edges in the current frame
            cv2.imshow("ProjectorOut", edges)
            pts = clean_points(extract_points_from_lines(edges))  # Clean extracted points
            
            # Detect candidate quadrilaterals from the edge points
            quads, num_quads = create_quadrilaterals(pts)  # Detect quadrilaterals
            if num_quads == 0:
                continue  # Skip frames with no valid quadrilaterals

            # Warp the texture onto each detected quadrilateral
            warped = [apply_homography(compute_homography_from_image(texture, q), frame.shape[1::-1], texture) for q in quads]
            montage = combine_images(warped)  # Combine the warped images into a montage
            if montage is None:
                continue

            # Project the combined image to the projector space using the homography matrix
            proj_img = cv2.warpPerspective(montage, H_cp, PROJ_SIZE)

            # Display the projected image on the projector
            cv2.imshow("ProjectorOut", proj_img)
            cv2.imshow("CameraView", frame)  # Display the original camera feed
            cv2.imshow("Edges", edges)  # Display the edges for debugging

            # Allow user to quit by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # If using a static image, display once and quit
            if USE_STATIC_IMAGE:
                cv2.waitKey(0)
                break
    finally:
        if cap is not None:
            cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()

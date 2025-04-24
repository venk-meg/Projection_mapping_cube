#!/usr/bin/env python3
import cv2, numpy as np, pathlib, time

# ---------- USER SETTINGS -------------------------------------------------
CAM_INDEX   = 1
PROJ_RES    = (1280, 720)
PROJ_X_OFFSET, PROJ_Y_OFFSET = 1920, 0
DICT_NAME   = cv2.aruco.DICT_5X5_100
BOARD_SIZE  = (5, 7)          # internal squares
SQUARE_LEN_U, MARKER_LEN_U = 1.0, 0.5
MIN_CAPTURES = 12
OUT_INTRINSICS, OUT_HOMOGRAPHY = "camera_intrinsics.npz", "H_cp.npy"
MIN_CORNERS_PER_FRAME, MIN_CORNERS_FOR_H = 8, 6
# -------------------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)
board = cv2.aruco.CharucoBoard(BOARD_SIZE, SQUARE_LEN_U, MARKER_LEN_U, aruco_dict)

# --- projector-side reference image & lookup -----------------------------
board_gray = board.generateImage(PROJ_RES)
board_bgr  = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)          # ▶ added
corn_p, ids_p, _ = cv2.aruco.detectMarkers(board_gray, aruco_dict)
ok_p, ch_p, ch_ids_p = cv2.aruco.interpolateCornersCharuco(corn_p, ids_p, board_gray, board)
if not ok_p or ch_ids_p is None:
    raise RuntimeError("Failed to detect corners in synthetic board image")

proj_corner_lookup = {int(ch_ids_p[i][0]): tuple(ch_p[i][0]) for i in range(len(ch_ids_p))}

# --- show board full-screen on projector ---------------------------------
cv2.namedWindow("ChArUco-Proj", cv2.WINDOW_NORMAL)
cv2.moveWindow("ChArUco-Proj", PROJ_X_OFFSET, PROJ_Y_OFFSET)
cv2.setWindowProperty("ChArUco-Proj", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("ChArUco-Proj", board_bgr)
cv2.waitKey(1)

print("[i]  Projected board ready.  Press c to capture, q to finish.\n")

detector_params = cv2.aruco.DetectorParameters()
all_charuco_corners, all_charuco_ids = [], []
cam_pts_for_H, proj_pts_for_H = [], []

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened(): raise IOError(f"Cannot open camera #{CAM_INDEX}")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        ok, ch_corners, ch_ids = False, None, None
        if ids is not None and len(ids):
            ok, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if ok and ch_ids is not None and len(ch_ids) >= 4:
                cv2.aruco.drawDetectedCornersCharuco(frame, ch_corners, ch_ids)

        cv2.imshow("Camera view", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            good_intr = ok and ch_ids is not None and len(ch_ids) >= MIN_CORNERS_PER_FRAME
            shared_ids = [int(cid) for cid in ch_ids.flatten() if cid in proj_corner_lookup] if ch_ids is not None else []
            good_H = len(shared_ids) >= MIN_CORNERS_FOR_H

            if good_intr:
                all_charuco_corners.append(ch_corners)
                all_charuco_ids.append(ch_ids)
                image_size = gray.shape[::-1]
                print(f"  • stored frame #{len(all_charuco_corners)} "
                      f"({len(ch_ids)} corners, {len(shared_ids)} shared)")
            else:
                print(f"  × skipped – only {0 if ch_ids is None else len(ch_ids)} corners")

            if good_H:
                for i, cid in enumerate(ch_ids.flatten()):
                    cid = int(cid)
                    if cid in proj_corner_lookup:
                        cam_pts_for_H.append(ch_corners[i][0])
                        proj_pts_for_H.append(proj_corner_lookup[cid])

        if key == ord('h') and good_H:               # ← new hot-key
            cam_pts_for_H = []                       # overwrite any previous pts
            proj_pts_for_H = []
            for i, cid in enumerate(ch_ids.flatten()):
                cid = int(cid)
                if cid in proj_corner_lookup:
                    cam_pts_for_H.append(ch_corners[i][0])
                    proj_pts_for_H.append(proj_corner_lookup[cid])
            print(f"  • selected this frame for homography ({len(cam_pts_for_H)} pts)")

        if key == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()

# ---------- INTRINSIC CALIBRATION ---------------------------------------
if len(all_charuco_corners) < MIN_CAPTURES:
    print(f"[!] Only {len(all_charuco_corners)} captures (need {MIN_CAPTURES}).")
    raise SystemExit

ret, K, dist, *_ = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners, all_charuco_ids, board, image_size, None, None)

print("\n=====  Intrinsic calibration  =====")
print(f"RMS reprojection error : {ret:.3f} px")
print("K:\n", K)
np.savez_compressed(OUT_INTRINSICS, K=K, dist=dist)
print(f"[i]  Saved intrinsics → {OUT_INTRINSICS}")

# ---------- HOMOGRAPHY ---------------------------------------------------
cam_pts  = np.asarray(cam_pts_for_H,  np.float32).reshape(-1,1,2)
proj_pts = np.asarray(proj_pts_for_H, np.float32).reshape(-1,1,2)

if len(cam_pts) < 4:
    print("[!] Need ≥4 shared points, got", len(cam_pts))
    raise SystemExit

cam_pts_u = cv2.undistortPoints(cam_pts, K, dist, P=K).reshape(-1,1,2)
H, mask   = cv2.findHomography(cam_pts_u, proj_pts, cv2.RANSAC, 3.0)
np.save(OUT_HOMOGRAPHY, H)

inliers = int(mask.sum())
print("\n=====  Camera → Projector homography  =====")
print(f"Inliers / total : {inliers} / {len(cam_pts)}")
proj_pred = cv2.perspectiveTransform(cam_pts_u, H)
err = np.linalg.norm(proj_pred - proj_pts, axis=2)
print(f"Mean reprojection err: {err.mean():.2f} px")

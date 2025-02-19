import cv2
import numpy as np
import glob
import os

def draw(img, corners, imgpts):
    corner_origin = tuple(corners[0].ravel().astype(int))
    x_axis = tuple(imgpts[0].ravel().astype(int))
    y_axis = tuple(imgpts[1].ravel().astype(int))
    z_axis = tuple(imgpts[2].ravel().astype(int))

    # Draw the axes
    img = cv2.line(img, corner_origin, x_axis, (0, 0, 255), 3)  # X - RED
    img = cv2.line(img, corner_origin, y_axis, (0, 255, 0), 3)  # Y - GREEN
    img = cv2.line(img, corner_origin, z_axis, (255, 0, 0), 3)  # Z - BLUE

    for corner in corners:
        x, y = corner.ravel()  # Flatten from (1,2) -> (2,)
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    return img

# -------------------------------------------------------------------
# Helper: Convert a 3x3 rotation + 3x1 translation to a 4x4 matrix
# -------------------------------------------------------------------
def rt_to_4x4(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t.ravel()
    return T

# -------------------------------------------------------------------
# Helper: Convert 4D homogeneous points (x, y, z, w) to 3D
# -------------------------------------------------------------------
def hom2cart(points_4d):
    # points_4d shape: (4, N)
    # return shape: (N, 3)
    points_4d /= points_4d[3, :]  # divide each column by w
    return points_4d[:3, :].T     # transpose so we get (N,3)


from scipy.optimize import minimize
def reprojection_error(params, points_3d, pts1_undist, pts2_undist, P1, P2):
    points_3d_opt = params[:len(points_3d) * 3].reshape(-1, 3)
    P1_opt = params[len(points_3d) * 3:len(points_3d) * 3 + 12].reshape(3, 4)
    P2_opt = params[len(points_3d) * 3 + 12:].reshape(3, 4)

    error = 0
    for i in range(len(points_3d_opt)):
        point_proj1 = P1_opt @ np.hstack([points_3d_opt[i], 1])
        point_proj1 = point_proj1[:2] / point_proj1[2]
        error += np.linalg.norm(point_proj1 - pts1_undist[i])
        point_proj2 = P2_opt @ np.hstack([points_3d_opt[i], 1])
        point_proj2 = point_proj2[:2] / point_proj2[2]
        error += np.linalg.norm(point_proj2 - pts2_undist[i])
    return error

def construct_3Dpoints(left_path, right_path, image_name, P1, P2, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, shift=0):
    pattern_size = (7, 6)

    # Load images
    left_image = os.path.join(left_path, image_name)
    right_image = os.path.join(right_path, image_name)
    img1 = cv2.imread(left_image)
    img2 = cv2.imread(right_image)

    # Convert to gray
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find corners
    found1, corners1 = cv2.findChessboardCornersSB(gray1, pattern_size)
    found2, corners2 = cv2.findChessboardCornersSB(gray2, pattern_size)

    if not (found1 and found2):
        print("Could not find corners in one or both images.")
        exit(1)

    # Optional: refine corners to subpixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)
    # corners1_subpix = cv2.find4QuadCornerSubpix(gray1, corners1, (5, 5))[1]
    # corners2_subpix = cv2.find4QuadCornerSubpix(gray2, corners2, (5, 5))[1]
    corners1_subpix = cv2.cornerSubPix(gray1, corners1, (3, 3), (-1, -1), criteria)
    corners2_subpix = cv2.cornerSubPix(gray2, corners2, (3, 3), (-1, -1), criteria)

    # print(corners1_subpix)
    # corners1_subpix is shape (N,1,2). We want shape (2,N) for triangulate
    pts1 = corners1_subpix.reshape(-1, 2).T+shift  # (2, N)
    pts2 = corners2_subpix.reshape(-1, 2).T+shift  # (2, N)

    pts1_undist = cv2.undistortPoints(pts1, cameraMatrixL, distCoeffsL, R=cameraMatrixL)
    pts2_undist = cv2.undistortPoints(pts2, cameraMatrixR, distCoeffsR, R=cameraMatrixR)

    # -------------------------------------------------------------------
    # 4. Triangulate points
    # -------------------------------------------------------------------
    points_4d = cv2.triangulatePoints(P1, P2, pts1_undist, pts2_undist)
    points_3d = hom2cart(points_4d)  # shape (N, 3)

    return points_3d

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
def calculate_plane_normal(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[-1]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    reference_point = np.array([0, 0, 1])  # Default reference point
    # Ensure the normal points towards the reference point
    if np.dot(normal_vector, reference_point - centroid) < 0:
        normal_vector = -normal_vector
    return normal_vector

def cal_rotation_translation(points_3d):
    pattern_size = (7, 6)
    square_size = 4 # mm
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    # objp shape => (42, 3)

    retval, M, inliers = cv2.estimateAffine3D(objp, points_3d)

    # M is a 3x4 matrix of the form [R | t], where R is 3x3, t is 3x1
    if retval:
        1
        # print("3D-3D estimation succeeded.")
        # print("Affine transform:\n", M)
    else:
        print("3D-3D estimation failed.")
        exit(1)

    # Convert to 4x4
    T_3d = np.eye(4, dtype=np.float64)
    T_3d[:3, :] = M
    # Now T_3d maps board coords to camera coords:
    #   X_cam = T_3d * [X_board, 1]^T
    # Extract R, t from T_3d
    R_est = T_3d[:3, :3]
    t_est = T_3d[:3, 3]
    R_est = M[:, :3]  # 3x3 rotation matrix
    t_est = M[:, 3]  # 3x1 translation vector
    # For convenience, convert R to a rotation vector (rvec) if desired
    rvec_est, _ = cv2.Rodrigues(R_est)

    inlier_objp = objp
    inlier_points_3d = points_3d
    # points_3d_n = calculate_plane_normal(inlier_points_3d)
    # objp_n = calculate_plane_normal(inlier_objp)

    objp_x = np.array([1, 0, 0])
    objp_n = np.array([0, 0, -1])
    points_3d_x = np.dot(R_est, np.array([1, 0, 0]))
    # points_3d_n = R_est @ np.array([0.0, 0.0, -1.0])
    points_3d_n = np.array([0.0, 0.0, -1.0])
    points_3d_n = calculate_plane_normal(inlier_points_3d)
    objp_n = calculate_plane_normal(inlier_objp)
    # version 1
    # t_center = (np.mean(inlier_points_3d, axis=0)-points_3d_n*9.46) - (np.mean(inlier_objp, axis=0)+objp_n*9.46)
    # version 2
    # t_center = (np.mean(inlier_points_3d, axis=0)-points_3d_n*28.5+points_3d_x*4.5) - (np.mean(inlier_objp, axis=0)-objp_n*28.5+objp_x*4.5)
    t_center = (np.mean(inlier_points_3d, axis=0)-points_3d_n*28.2+points_3d_x*4.2) - (np.mean(inlier_objp, axis=0)+objp_n*28.2+objp_x*4.2)
    # t_center = (np.mean(inlier_points_3d, axis=0)) - (np.mean(inlier_objp, axis=0))
    print("normal vector....", points_3d_n, objp_n, points_3d_x)

    # print("Estimated rotation matrix:\n", R_est)
    # print("Estimated translation vector:", t_est)
    # print("Rotation vector (Rodrigues):", rvec_est.ravel())
    return rvec_est, t_est, t_center

def camera_cali(image_folder="", patter_size=(7, 6)):
    # Prepare object points (3D coordinates of chessboard corners)
    # objp = np.zeros((6*7,3), np.float32)
    # objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) *20 # unit mm
    # objp = np.zeros((12 * 15, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:15, 0:12].T.reshape(-1, 2) * 12  # unit mm
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 12  # unit mm

    # Arrays to store object points and image points
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Loop through calibration images
    images = os.listdir(image_folder)
    for fname in images:
        img = cv2.imread(os.path.join(image_folder, fname))
        # img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        # ret, corners = cv2.findChessboardCornersSB(gray, (7, 6), None)
        # ret, corners = cv2.findChessboardCornersSB(gray, (15, 12), None)
        ret, corners = cv2.findChessboardCornersSB(gray, (9, 6), None)
        print(fname, ret)
        if ret:
            # corners_subpix = cv2.find4QuadCornerSubpix(gray, corners, (11, 11))[1]
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_subpix)

    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist)
    return ret, mtx, dist

def stereoCali(left_path, right_path, mtxL, distL, mtxR, distR):
    # pattern_size = (7, 6)
    # square_size = 20
    # pattern_size = (15, 12)
    pattern_size = (9, 6)
    square_size = 12

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # We'll collect these for stereoCalibrate
    objpoints_stereo = []
    imgpoints_left = []
    imgpoints_right = []

    # Suppose your image pairs are named left01.jpg & right01.jpg, left02.jpg & right02.jpg, etc.
    # Or you have a list of matched pairs.
    left_images = sorted(os.listdir(left_path))
    right_images = sorted(os.listdir(right_path))
    # print(left_images, right_images)
    for fnameL, fnameR in zip(left_images, right_images):
        imgL = cv2.imread(os.path.join(left_path, fnameL))
        imgR = cv2.imread(os.path.join(right_path, fnameR))
        # imgL = cv2.convertScaleAbs(imgL, alpha=1.1, beta=10)
        # imgR = cv2.convertScaleAbs(imgR, alpha=1.1, beta=10)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCornersSB(grayL, pattern_size)
        retR, cornersR = cv2.findChessboardCornersSB(grayR, pattern_size)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

        print(fnameL, retL, retR)
        if retL and retR:
            # Refine
            cornersL_subpix = cv2.cornerSubPix(grayL, cornersL, (3, 3), (-1, -1), criteria)
            cornersR_subpix = cv2.cornerSubPix(grayR, cornersR, (3, 3), (-1, -1), criteria)
            # cornersL_subpix = cv2.find4QuadCornerSubpix(grayL, cornersL, (11, 11))[1]
            # cornersR_subpix = cv2.find4QuadCornerSubpix(grayR, cornersR, (11, 11))[1]

            # Save valid data
            objpoints_stereo.append(objp)
            imgpoints_left.append(cornersL_subpix)
            imgpoints_right.append(cornersR_subpix)

    # We assume we already have single-camera calibration results for each camera:
    # mtxL, distL, mtxR, distR  (from the individual calibration step)

    # Stereo Calibration
    flags = 0
    # You can add flags like cv2.CALIB_FIX_INTRINSIC if you want to fix intrinsics
    # or let stereoCalibrate refine them slightly.

    ret, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = cv2.stereoCalibrate(
        objpoints_stereo,  # list of 3D points in real world
        imgpoints_left,  # 2D points in left camera
        imgpoints_right,  # 2D points in right camera
        mtxL, distL,  # left intrinsics
        mtxR, distR,  # right intrinsics
        grayL.shape[::-1],  # image size
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("Stereo Calibration done, RMS error =", ret)
    print("Left camera matrix:\n", cameraMatrixL)
    print("Right camera matrix:\n", cameraMatrixR)
    print("Rotation between cameras:\n", R)
    print("Translation between cameras:\n", T)
    return ret, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,  R, T


def rt_to_matrix(rvec, tvec):
    """Convert rvec, tvec to a 4x4 homogeneous transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)  # 3x3 rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T

from scipy.spatial.transform import Rotation as RR
def rvec_to_unit_vector(rvec, d=[0, 0, -1]):
    rvec = rvec.flatten()
    r = RR.from_rotvec(rvec)
    direction = r.apply(d)
    return direction / np.linalg.norm(direction)
    # rvec = rvec.flatten()
    # theta = np.linalg.norm(rvec)
    # # Normalize the rotation vector
    # if theta != 0:
    #     k = rvec / theta
    # else:
    #     return np.array([0, 0, 1])  # Default forward direction
    #
    # # Compute the rotation matrix using Rodrigues' formula
    # K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    # R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    #
    # # Extract the direction vector (third column of R)
    # direction = R[:, 2]
    # return direction

def relative_pose(rvec_ref, tvec_ref, rvec, tvec):
    """
    Given a reference pose (rvec_ref, tvec_ref) and another pose (rvec, tvec),
    return the pose of 'rvec, tvec' relative to the reference.
    """
    T_ref = rt_to_matrix(rvec_ref, tvec_ref)  # 4x4
    T = rt_to_matrix(rvec, tvec)             # 4x4

    # Compute relative transform T_rel = inv(T_ref) * T
    T_rel = np.linalg.inv(T_ref) @ T

    # Convert back to rvec, tvec
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    rvec_rel, _ = cv2.Rodrigues(R_rel)

    unit_ref = rvec_to_unit_vector(rvec_ref)
    unit_v = rvec_to_unit_vector(rvec)
    dot_product = np.dot(unit_ref, unit_v)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_rel = np.degrees(angle_rad)

    return angle_rel, rvec_rel, t_rel

if __name__ == "__main__":
    # single camera calibration
    # left_path = "./test/cam2_calibration"   # data for v1
    # right_path = "./test/cam1_calibration"  # data for v1
    v = "v26"
    left_path = "../data/Images_Sensors_0209/calibration_{}/cam1".format(v)
    right_path = "../data/Images_Sensors_0209/calibration_{}/cam2".format(v)
    ret1, K1, d1 = camera_cali(image_folder=left_path)
    ret2, K2, d2 = camera_cali(image_folder=right_path)

    ret, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T = stereoCali(left_path, right_path, K1, d1, K2, d2)
    # -------------------------------------------------------------------
    # 2. Build Projection Matrices (P1, P2)
    #    For triangulation, you can rectify or just do direct
    #    P = K [ R | T ]. If R=I, T=0 for the first camera.
    # -------------------------------------------------------------------
    # Camera1 is reference, so:
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = cameraMatrixL @ np.hstack([R1, t1])
    # Camera2 relative to Camera1
    P2 = cameraMatrixR @ np.hstack([R, T.reshape(3, 1)])
    print(R, T)
    # save P1, P2
    np.savez('./stereoCamera_P1P2_{}.npz'.format(v), P1=P1, P2=P2, R=R, T=T, cL=cameraMatrixL, dL=distCoeffsL, cR=cameraMatrixR, dR=distCoeffsR)
    data = np.load('./stereoCamera_P1P2_v8.npz')
    P1 = data['P1']
    P2 = data['P2']
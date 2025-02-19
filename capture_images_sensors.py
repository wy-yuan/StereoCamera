import numpy as np
import cv2
import glob
import os
from datetime import datetime
import time
import serial
import shutil
# from stereoCamera import construct_3Dpoints, cal_rotation_translation, relative_pose, rvec_to_unit_vector
from stereoCamera import *
import pickle
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def capture_images_sensors(output_dir, duration=60, frame_number = 0):
    # initialize sensors
    arduino = serial.Serial('COM4', 9600, timeout=1)
    arduino.reset_input_buffer()
    time.sleep(5)

    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is usually the default USB camera
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    for i in range(20):
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()

    if not cap.isOpened():
        print("Error: Could not open camera 1.")
        return
    if not cap1.isOpened():
        print("Error: Could not open camera 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    exposure_value = -4  # Adjust this value as needed
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 50)  # 80 corresponds to 200mm

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap1.set(cv2.CAP_PROP_FPS, 30)
    # # cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # # cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap1.set(cv.CAP_PROP_AUTO_WB, 0)
    exposure_value = -4  # Adjust this value as needed
    cap1.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap1.set(cv2.CAP_PROP_FOCUS, 65)  # 80 corresponds to 200mm

    time.sleep(2)
    start_time = time.time()
    while time.time() - start_time < duration:
        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()
            sensors = [float(x) for x in data.split(",")[:-1]]

            ret, frame = cap.read()
            ret1, frame1 = cap1.read()

            if (not ret) or (not ret1) or (len(sensors) != 12):
                print("Failed to grab frames or sensors", ret, ret1, len(sensors))
                continue
            else:
                frame_number += 1
                sensors = np.array(sensors)
                sensors[[0, 3, 6, 9]] = -sensors[[0, 3, 6, 9]]  # X values should reverse the sign
                np.save(os.path.join(output_dir, "sensor", f"{frame_number}.npy"), sensors)
                filename1 = os.path.join(output_dir, "cam1", f"{frame_number}.jpg")
                filename2 = os.path.join(output_dir, "cam2", f"{frame_number}.jpg")
                cv2.imwrite(filename1, frame)
                cv2.imwrite(filename2, frame1)
    cap.release()
    cap1.release()
    print("Frame number:", frame_number)


def capture_images_withKeys(output_dir, frame_number = 0):
    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is usually the default USB camera
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    for i in range(20):
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()

    if not cap.isOpened():
        print("Error: Could not open camera 1.")
        return
    if not cap1.isOpened():
        print("Error: Could not open camera 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    exposure_value = -4  # Adjust this value as needed
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 50)  # 80 corresponds to 200mm

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap1.set(cv2.CAP_PROP_FPS, 30)
    # # cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # # cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap1.set(cv.CAP_PROP_AUTO_WB, 0)
    exposure_value = -4  # Adjust this value as needed
    cap1.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap1.set(cv2.CAP_PROP_FOCUS, 60)  # 80 corresponds to 200mm

    time.sleep(2)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
        cv2.imshow('img', cv2.resize(cv2.hconcat([frame, frame1]), (int(frame.shape[1] / 1), int(frame.shape[0] / 2))))
        if (not ret) or (not ret1):
            print("Failed to grab frames or sensors", ret, ret1)
            continue
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 27 is the ASCII code for Esc
                break
            elif key == ord('s'):  # 's' key to save the frame
                frame_number += 1
                print("Frame number:", frame_number)
                filename1 = os.path.join(output_dir, "cam1", f"{frame_number}.jpg")
                filename2 = os.path.join(output_dir, "cam2", f"{frame_number}.jpg")
                cv2.imwrite(filename1, frame)
                cv2.imwrite(filename2, frame1)
    cap.release()
    cap1.release()
    print("Frame number:", frame_number)

def capture_images_sensor_withKeys(output_dir, frame_number = 0):
    # initialize sensors
    arduino = serial.Serial('COM4', 9600, timeout=1)
    arduino.reset_input_buffer()
    time.sleep(5)

    # Initialize the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is usually the default USB camera
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    for i in range(20):
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()

    if not cap.isOpened():
        print("Error: Could not open camera 1.")
        return
    if not cap1.isOpened():
        print("Error: Could not open camera 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    exposure_value = -4  # Adjust this value as needed
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 50)  # 80 corresponds to 200mm

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap1.set(cv2.CAP_PROP_FPS, 30)
    # # cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # # cap1.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cap1.set(cv.CAP_PROP_AUTO_WB, 0)
    exposure_value = -4  # Adjust this value as needed
    cap1.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap1.set(cv2.CAP_PROP_FOCUS, 60)  # 80 corresponds to 200mm

    time.sleep(2)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
        cv2.imshow('img', cv2.resize(cv2.hconcat([frame, frame1]), (int(frame.shape[1] / 1), int(frame.shape[0] / 2))))
        if (not ret) or (not ret1):
            print("Failed to grab frames or sensors", ret, ret1)
            continue
        else:
            key = cv2.waitKey(1) & 0xFF
            if arduino.in_waiting > 0:
                data = arduino.readline().decode('utf-8').strip()
                sensors = [float(x) for x in data.split(",")[:-1]]

            if key == 27:  # 27 is the ASCII code for Esc
                break
            elif key == ord('s'):  # 's' key to save the frame
                if len(sensors)==12:
                    frame_number += 1
                    sensors = np.array(sensors)
                    sensors[[0, 3, 6, 9]] = -sensors[[0, 3, 6, 9]]  # X values should reverse the sign
                    np.save(os.path.join(output_dir, "sensor", f"{frame_number}.npy"), sensors)
                    print("Frame number:", frame_number)
                    filename1 = os.path.join(output_dir, "cam1", f"{frame_number}.jpg")
                    filename2 = os.path.join(output_dir, "cam2", f"{frame_number}.jpg")
                    cv2.imwrite(filename1, frame)
                    cv2.imwrite(filename2, frame1)

    cap.release()
    cap1.release()
    print("Frame number:", frame_number)


def assess_chessboard_quality(image_path, board_size=(7, 6)):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCornersSB(gray, board_size, None)

    if ret:
        # Refine corner detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        x_min, y_min = np.min(corners2[:, 0, :], axis=0).astype(int)
        x_max, y_max = np.max(corners2[:, 0, :], axis=0).astype(int)
        # Extract the chessboard region
        chessboard_region = gray[y_min:y_max, x_min:x_max]
        # Calculate the Laplacian of the extracted chessboard region
        laplacian = cv2.Laplacian(chessboard_region, cv2.CV_64F)
        sharpness = laplacian.var()
        # Calculate the width and height of the bounding box
        # width, height = x_max - x_min, y_max - y_min

        # Calculate reprojection error
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        _, rvec, tvec = cv2.solvePnP(objp, corners2, np.eye(3), None)
        imgpoints, _ = cv2.projectPoints(objp, rvec, tvec, np.eye(3), None)
        error = cv2.norm(corners2, imgpoints, cv2.NORM_L2) / len(imgpoints)

        return True, len(corners2), sharpness, error
    else:
        return False, 0, 0, np.inf

def filter_image(cam1_dir, cam2_dir, filter_cam1_dir, filter_cam2_dir):
    image_name_list = os.listdir(cam1_dir)
    for image_name in image_name_list:
        image_number = int(image_name.split(".")[0])
        if image_number<40 or (not image_name.endswith(".jpg")):
            continue
        # print(image_name)
        cam1_image = os.path.join(cam1_dir, image_name)
        quality_file_name = os.path.join(cam1_dir, image_name.replace(".jpg", ".pkl"))
        if os.path.exists(quality_file_name):
            with open(quality_file_name, 'rb') as f:
                qua = pickle.load(f)
            ret, corner_num, sharpness, reproj_error = qua['ret'], qua['corner_num'], qua['sharpness'], qua['reproj_error']
        else:
            ret, corner_num, sharpness, reproj_error = assess_chessboard_quality(cam1_image)
            qua = {'ret': ret, 'corner_num': corner_num, 'sharpness': sharpness, 'reproj_error': reproj_error}
            with open(quality_file_name, 'wb') as f:
                pickle.dump(qua, f)

        if ret and corner_num == 42 and sharpness > sharpness_th and reproj_error < reproj_error_th:
            shutil.copy(cam1_image, filter_cam1_dir)

        cam2_image = os.path.join(cam2_dir, image_name)
        quality_file_name = os.path.join(cam2_dir, image_name.replace(".jpg", ".pkl"))
        if os.path.exists(quality_file_name):
            with open(quality_file_name, 'rb') as f:
                qua = pickle.load(f)
                ret, corner_num, sharpness, reproj_error = qua['ret'], qua['corner_num'], qua['sharpness'], qua[
                    'reproj_error']
        else:
            ret, corner_num, sharpness, reproj_error = assess_chessboard_quality(cam2_image)
            qua = {'ret': ret, 'corner_num': corner_num, 'sharpness': sharpness, 'reproj_error': reproj_error}
            with open(quality_file_name, 'wb') as f:
                pickle.dump(qua, f)
        if ret and corner_num == 42 and sharpness > sharpness_th and reproj_error < reproj_error_th:
            shutil.copy(cam2_image, filter_cam2_dir)

def cal_position_orientation(left_path, right_path, ref_left_path, ref_right_path, filter_pos_ori_dir, stereoP_file):
    data = np.load(stereoP_file)
    P1 = data['P1']
    P2 = data['P2']
    cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR = data['cL'], data['dL'], data['cR'], data['dR']

    # calculate reference rotation and translation
    # ref_left_path, ref_right_path = "./test0121/cam2", "./test0121/cam1"
    points_3d_1 = construct_3Dpoints(ref_left_path, ref_right_path, "14.jpg", P1, P2, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, shift=0)
    # points_3d_1 = []
    # for id in range(1, 11):
    #     points_3d_id = construct_3Dpoints(ref_left_path, ref_right_path, "{}.jpg".format(id), P1, P2, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, shift=0)
    #     points_3d_1.append(points_3d_id)
    # print(np.array(points_3d_1).shape)
    # points_3d_1 = np.mean(np.array(points_3d_1), axis=0)
    rvecs_1, tvecs_1, t_c_1 = cal_rotation_translation(points_3d_1)
    # print("Reference:", rvecs_1, tvecs_1)

    # pick images in both paths
    left_list = os.listdir(left_path)
    right_list = os.listdir(right_path)
    image_list = sorted(list(set(left_list) & set(right_list)))
    for image_name in image_list:
        pose_file_name = os.path.join(filter_pos_ori_dir, image_name.replace(".jpg", ".npy"))
        image_number = int(image_name.split(".")[0])
        # print("debug......", image_number)
        if os.path.exists(pose_file_name) or (image_number<=60):
            continue
        else:
            points_3d = construct_3Dpoints(left_path, right_path, image_name, P1, P2, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR)
            rvecs, tvecs, t_c = cal_rotation_translation(points_3d)
            ang_rel, rvec_rel, tvec_rel = relative_pose(rvecs_1, t_c_1, rvecs, t_c)
            # tvec_rel = t_c - t_c_1
            unit_vector = rvec_to_unit_vector(rvec_rel, d=[-1, 0, 0])  # [0, 0, -1]
            tvec_rel_ = tvec_rel[1], tvec_rel[0], -tvec_rel[2]
            unit_vector_ = unit_vector[1], unit_vector[0], -unit_vector[2]
            magnet_pose = np.hstack([tvec_rel_, unit_vector_])
            print(image_name, magnet_pose)
            np.save(pose_file_name, magnet_pose)


def cal_chessboard_rotation_translation(points_3d, p_n, o_n):
    pattern_size = (7, 6)
    square_size = 4 # mm
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    # objp shape => (42, 3)

    # At this point, we have:
    #  - points_3d: the set of 3D corners in camera1 frame
    #  - objp: the same corners in the chessboard's local (board) frame

    # -------------------------------------------------------------------
    # 6. Estimate chessboard pose from the 3D-3D correspondence
    #    We want a transform T that maps board-frame points -> camera1-frame points
    #    One approach: cv2.estimateAffine3D (if scale is known, it's an alignment)
    #    Or do a custom SVD-based solve. We'll do estimateAffine3D for simplicity.
    # -------------------------------------------------------------------
    # Make sure they're the same ordering of corners:
    # points_3d shape: (42, 3)
    # objp shape: (42, 3)
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

    # For convenience, convert R to a rotation vector (rvec) if desired
    rvec_est, _ = cv2.Rodrigues(R_est)

    # use centroid of the chessboard corner as translation reference points
    # t_center = np.mean(points_3d, axis=0) - np.mean(objp, axis=0)

    # use centroid of the magnet as translation reference points
    # inlier_objp = objp[inliers.ravel() == 1]
    # inlier_points_3d = points_3d[inliers.ravel() == 1]
    inlier_objp = objp
    inlier_points_3d = points_3d
    points_3d_n = calculate_plane_normal(inlier_points_3d)
    objp_n = calculate_plane_normal(inlier_objp)
    objp_x = np.array([1, 0, 0])
    objp_n = np.array([0, 0, -1])
    points_3d_x = np.dot(R_est, np.array([1, 0, 0]))
    points_3d_n = np.dot(R_est, np.array([0, 0, -1]))
    # t_center = (np.mean(inlier_points_3d, axis=0)-p_n*9.46) - (np.mean(inlier_objp, axis=0)-o_n*9.46)
    t_center = (np.mean(inlier_points_3d, axis=0) - points_3d_n * 28.5 + points_3d_x * 4.5) - (
                np.mean(inlier_objp, axis=0) + objp_n * 28.5 + objp_x * 4.5)

    # print("Estimated rotation matrix:\n", R_est)
    # print("Estimated translation vector:", t_est)
    # print("Rotation vector (Rodrigues):", rvec_est.ravel())
    return rvec_est, t_est, t_center

def cal_ref_rotation_translation(objp, points_3d):
    # objp shape => (27, 3)
    retval, M, inliers = cv2.estimateAffine3D(objp, points_3d)
    # M is a 3x4 matrix of the form [R | t], where R is 3x3, t is 3x1
    if retval:
        1
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

    # For convenience, convert R to a rotation vector (rvec) if desired
    rvec_est, _ = cv2.Rodrigues(R_est)

    xx = np.array([points_3d[6, :] - points_3d[0, :], points_3d[8, :] - points_3d[2, :], points_3d[15, :] - points_3d[9, :],
                   points_3d[17, :] - points_3d[11, :], points_3d[24, :] - points_3d[18, :],
                   points_3d[7, :] - points_3d[1, :], points_3d[16, :] - points_3d[10, :],
                   points_3d[25, :] - points_3d[19, :],
              points_3d[26, :] - points_3d[20, :]])
    # xx = np.array([points_3d[7, :] - points_3d[1, :], points_3d[16, :] - points_3d[10, :], points_3d[25, :] - points_3d[19, :]])
    points_3d_n = np.mean(xx, axis=0)
    points_3d_n = points_3d_n / np.linalg.norm(points_3d_n)
    yy = np.array([objp[6,:]-objp[0,:], objp[8,:]-objp[2,:],
                              objp[15, :] - objp[9, :], objp[17, :] - objp[11, :],
                   objp[7, :] - objp[1, :], objp[16, :] - objp[10, :], objp[25, :] - objp[19, :],
                              objp[24,:]-objp[18,:], objp[26,:]-objp[20,:]])
    # yy = np.array([objp[7, :] - objp[1, :], objp[16, :] - objp[10, :], objp[25, :] - objp[19, :]])
    objp_n = np.mean(yy, axis=0)
    objp_n = objp_n / np.linalg.norm(objp_n)
    print("debug", objp_n, points_3d_n)
    t_center = (np.mean(points_3d, axis=0)-points_3d_n*9.46) - (np.mean(objp, axis=0)-objp_n*9.46)
    # t_center = (points_3d[13,:]-points_3d_n*9.46) - (objp[13,:]-objp_n*9.46)
    objp_x = np.array([1, 0, 0])
    objp_n = np.array([0, 0, -1])
    points_3d_x = np.dot(R_est, np.array([1, 0, 0]))
    points_3d_n = np.dot(R_est, np.array([0, 0, -1]))
    t_center = (np.mean(points_3d, axis=0) - points_3d_n * 28.5 + points_3d_x * 4.5) - (
            np.mean(objp, axis=0) - objp_n * 28.5 + objp_x * 4.5)
    return rvec_est, t_est, t_center, points_3d_n, objp_n


def cal_position_orientation_error(pos, ori, pre_pos, pre_ori):
    pos_error = np.linalg.norm(pos-pre_pos)*1000
    v1 = ori / np.linalg.norm(ori)
    v2 = pre_ori / np.linalg.norm(pre_ori)
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    angle_degrees = np.degrees(angle)
    return pos_error, angle_degrees


def rotate_vector_around_y(vector, angle_deg):
    """
    Rotate a 3D vector around the Y-axis by a given angle (in degrees).
    """
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)
    # Define the rotation matrix for the Y-axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    R_y = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    # Apply the rotation to the vector
    rotated_vector = R_y @ vector
    return rotated_vector

def rotate_vector_around_x(vector, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])
    return np.dot(rotation_matrix, vector)

def get_filteredSensor_magnet_pose(filter_pos_ori_dir, sensor_dir, csv_dir, bias_file, data_len=8):
    bias = np.load(bias_file)
    num_list = sorted([int(filename.split(".")[0]) for filename in os.listdir(filter_pos_ori_dir)])
    print(num_list)
    data_points_list = []
    data_list = []
    i = 0
    magnet_pose = np.load(os.path.join(filter_pos_ori_dir, "{}.npy".format(num_list[0])))
    pre_pos = magnet_pose[:3] / 1000 + np.array([0, 0, 0.15])
    pre_ori = -magnet_pose[3:]
    sensor = np.load(os.path.join(sensor_dir, "{}.npy".format(num_list[0]))) - bias
    data_list.append(np.hstack([sensor, pre_pos, pre_ori]))
    # print(data_list)
    i = i + 1
    while i < len(num_list):
        num = num_list[i]
        magnet_pose = np.load(os.path.join(filter_pos_ori_dir, "{}.npy".format(num)))
        pos = magnet_pose[:3] / 1000 + np.array([0, 0, 0.15])
        ori = -magnet_pose[3:]
        pos_diff, ori_diff = cal_position_orientation_error(pos, ori, pre_pos, pre_ori)
        sensor = np.load(os.path.join(sensor_dir, "{}.npy".format(num))) - bias
        # print(pos_diff, ori_diff)
        if pos_diff < 0.3 and ori_diff < 0.3:
            data_list.append(np.hstack([sensor, pre_pos, pre_ori]))
        else:
            if len(data_list) >= data_len:
                point = np.mean(np.array(data_list), axis=0)
                print("point:", point)
                data_points_list.append(point)
            data_list=[]
            data_list.append(np.hstack([sensor, pre_pos, pre_ori]))
        pre_pos, pre_ori = pos, ori
        i = i + 1
    if len(data_list) >= data_len:
        point = np.mean(np.array(data_list), axis=0)
        print("point:", point)
        data_points_list.append(point)

    to_be_saved = np.array(data_points_list)
    dataTable = pd.DataFrame(to_be_saved, columns=[
        'Sen0x', 'Sen0y', 'Sen0z', 'Sen1x', 'Sen1y', 'Sen1z', 'Sen2x', 'Sen2y', 'Sen2z',
        'Sen3x', 'Sen3y', 'Sen3z', 'MagPx', 'MagPy', 'MagPz', 'MagOx', 'MagOy', 'MagOz'])
    dataTable.to_csv(os.path.join(csv_dir, "NoNormalization.csv"), index=False)

    B_m = np.load(os.path.join("../data/simulated_data_sensorR5cm_212121_normB_dedup_Ori26/", 'normB_parameters.npy'),
                  allow_pickle=True).item()
    B_mean, B_min, B_max = B_m['B_mean'], B_m['B_min'], B_m['B_max']
    to_be_saved[:, :12] = (to_be_saved[:, :12] - B_mean) / (B_max - B_min)
    dataTable = pd.DataFrame(to_be_saved, columns=[
        'Sen0x', 'Sen0y', 'Sen0z', 'Sen1x', 'Sen1y', 'Sen1z', 'Sen2x', 'Sen2y', 'Sen2z',
        'Sen3x', 'Sen3y', 'Sen3z', 'MagPx', 'MagPy', 'MagPz', 'MagOx', 'MagOy', 'MagOz'])
    dataTable.to_csv(os.path.join(csv_dir, "Normalization.csv"), index=False)

def characterization(filter_pos_ori_dir, save_name):
    # create gt matrix
    pos_gt = []
    for y in [-50, -25, 0, 25, 50]:
        for z in [-50, 0, 50]:
            for x in [-50, 0, 50]:
                pos_gt.append([x, y, z])
    pos_gt = np.array(pos_gt)

    num_list = sorted([int(filename.split(".")[0]) for filename in os.listdir(filter_pos_ori_dir)])
    # print(num_list)
    magnet_pos = []
    for i in range(len(num_list)):
        magnet_pose = np.load(os.path.join(filter_pos_ori_dir, "{}.npy".format(num_list[i])))
        magnet_pos.append(magnet_pose)

    magnet_pos = np.array(magnet_pos)
    # print(magnet_pos.shape)
    # magnet_pos[6, :3] = pos_gt[6, :3]
    pos_error = np.linalg.norm(magnet_pos[:, :3] - pos_gt[:, :3], axis=1)
    pos_error_x = magnet_pos[:, 0] - pos_gt[:, 0]
    pos_error_y = magnet_pos[:, 1] - pos_gt[:, 1]
    pos_error_z = magnet_pos[:, 2] - pos_gt[:, 2]
    np.set_printoptions(precision=3, suppress=True)
    print("Overall: ", np.mean(abs(pos_error)), np.max(abs(pos_error)))
    print(pos_error)
    print("\n X Error: ", np.mean(abs(pos_error_x)), np.max(abs(pos_error_x)))
    print(pos_error_x)
    print("\n Y Error: ", np.mean(abs(pos_error_y)), np.max(abs(pos_error_y)))
    print(pos_error_y)
    print("\n Z Error: ", np.mean(abs(pos_error_z)), np.max(abs(pos_error_z)))
    print(pos_error_z)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    fig.add_trace(go.Scatter3d(x=magnet_pos[:, 0], y=magnet_pos[:, 1], z=magnet_pos[:, 2],
                mode='markers',marker=dict(size=5,color=1,)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=pos_gt[:, 0], y=pos_gt[:, 1], z=pos_gt[:, 2],
            mode='markers', marker=dict(size=3, color=2, )), row=1, col=1)
    fig.update_layout(title='',height=1000,width=1600, scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z',
            xaxis=dict(range=[-60, 60]),yaxis=dict(range=[-60, 60]),zaxis=dict(range=[-60, 60])),)
    fig.write_html("../figures/camera_validation/{}.html".format(save_name))

def cal_cameras_RT(v):
    version = v
    data = np.load('./stereoCamera_P1P2_{}.npz'.format(version))
    R, T = data['R'], data['T']
    theta = np.arccos((np.trace(R) - 1) / 2)
    # Convert to degrees
    angle_deg = np.degrees(theta)
    print(version, angle_deg, T.flatten())


if __name__ == "__main__":
    # sharpness_th, reproj_error_th = 1000, 0.3
    sharpness_th, reproj_error_th = 10, 5
    sharpness_th, reproj_error_th = 800, 0.3
    # output_dir = "./Images_Sensors_0122/Images_Sensors_0122_validation/"
    # output_dir = "../data/Images_Sensors_0129/characterization_0203_10"
    v = "v26"
    save_name = "{}_y_6".format(v)
    ref_name = "{}_ref".format(v)  # "{}_1".format(v)
    output_dir = "../data/Images_Sensors_0213/data_{}/".format(save_name)
    # output_dir = "../data/Images_Sensors_0209/calibration_v26/"
    stereoP_file = './stereoCamera_P1P2_{}.npz'.format(v)
    # ref_left_path, ref_right_path = "../data/Images_Sensors_0129/ref/cam1", "../data/Images_Sensors_0129/ref/cam2"
    ref_left_path, ref_right_path = "../data/Images_Sensors_0210/data_{}/cam1".format(ref_name), "../data/Images_Sensors_0210/data_{}/cam2".format(ref_name)
    # output_dir = "./test0121/"
    os.makedirs(output_dir, exist_ok=True)
    cam1_dir = os.path.join(output_dir, "cam1")
    cam2_dir = os.path.join(output_dir, "cam2")
    filter_cam1_dir = os.path.join(output_dir, "filter_cam1")
    filter_cam2_dir = os.path.join(output_dir, "filter_cam2")
    filter_pos_ori_dir = os.path.join(output_dir, "filter_pos_ori")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)
    os.makedirs(filter_cam1_dir, exist_ok=True)
    os.makedirs(filter_cam2_dir, exist_ok=True)
    os.makedirs(filter_pos_ori_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Step 1 get calibrated parameters
    cal_cameras_RT(v)
    # Step 2 capture chessboard images
    # capture_images_sensors(output_dir, duration=600, frame_number=0)
    # capture_images_withKeys(output_dir, frame_number=0)
    capture_images_sensor_withKeys(output_dir, frame_number=0)

    # Step 3 calculate position and orientation
    # cal_position_orientation(filter_cam1_dir, filter_cam2_dir, ref_left_path, ref_right_path, filter_pos_ori_dir, stereoP_file)
    cal_position_orientation(cam1_dir, cam2_dir, ref_left_path, ref_right_path, filter_pos_ori_dir, stereoP_file)





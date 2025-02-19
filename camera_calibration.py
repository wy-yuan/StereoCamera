# The calibration process is based on the pinhole camera model:
# Pinhole Camera Model:
#
# s * [u]   [fx  0  cx] [r11 r12 r13 t1] [X]
#     [v] = [ 0 fy  cy] [r21 r22 r23 t2] [Y]
#     [1]   [ 0  0   1] [r31 r32 r33 t3] [Z]
#                                        [1]
#
# Where:
# (u, v) are image coordinates
# (X, Y, Z) are world coordinates
# (fx, fy) are focal lengths
# (cx, cy) is the principal point
# [rij] is the rotation matrix
# [ti] is the translation vector
# s is a scale factor

import cv2
import numpy as np
from datetime import datetime
import time
import os

def capture_images_with_timestamps(output_dir, capture_interval=5, duration=60):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default USB camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # exposure_value = -30  # Adjust this value as needed
    # cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 80)  # 80 corresponds to 200mm

    start_time = time.time()
    frame_count = 0

    while (time.time() - start_time) < duration:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Save the frame as an image with timestamp in the filename
            filename = f"{output_dir}/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            print(f"Saved: {filename}")

            frame_count += 1

            # Wait for the specified interval
            time.sleep(capture_interval)
        else:
            print("Error: Could not read frame.")
            break

    # Release the camera
    cap.release()
    print(f"Captured {frame_count} images.")

def camera_cali(image_folder="", patter_size=(7, 6)):
    # Prepare object points (3D coordinates of chessboard corners)
    objp = np.zeros((6*7,3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  # unit mm

    # Arrays to store object points and image points
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Loop through calibration images
    images = os.listdir(image_folder)
    for fname in images:
        print(fname)
        img = cv2.imread(os.path.join(image_folder, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        print(ret)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist)
    # print(rvecs, tvecs)



if __name__ == "__main__":
    output_directory = "./captured_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # capture_images_with_timestamps(output_directory, capture_interval=0.5, duration=2)

    camera_cali(image_folder="./captured_images/test/cam1_calibration")

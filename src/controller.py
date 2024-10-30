from PyQt6 import QtWidgets, QtGui, QtCore

import cv2
import os
import numpy as np

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.set_parameters()
    
    def set_parameters(self):
        self.folder_path = 'C:/Users/sean/Desktop/cvdl_hw1/resources/Dataset_CvDl_Hw1/Q1_Image'
        self.corners = []
        self.h = None
        self.w = None
        self.rvecs = None
        self.tvecs = None
        self.dist = None

    def setup_control(self):
        self.ui.load_folder_button.clicked.connect(self.load_folder_button_clicked)
        self.ui.load_image_l_button.clicked.connect(self.load_image_l_button_clicked)
        self.ui.load_image_r_button.clicked.connect(self.load_image_r_button_clicked)
        self.ui.find_corners_button.clicked.connect(self.find_corners_button_clicked)
        self.ui.find_intrinsic_button.clicked.connect(self.find_intrinsic_button_clicked)
        self.ui.find_extrinsic_button.clicked.connect(self.find_extrinsic_button_clicked)
        self.ui.find_distortion_button.clicked.connect(self.find_distortion_button_clicked)

    def load_folder_button_clicked(self):
        """
        Use QFileDialog to open a folder selection dialog and get the selected folder path.
        """
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder", "")
        print(self.folder_path)

    def load_image_l_button_clicked(self):
        """
        Use QFileDialog to open a file selection dialog and get the selected file path.
        """
        file_path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "Select a left image", self.folder_path, "Image files (*.bmp)")
        print(file_path)
        pass

    def load_image_r_button_clicked(self):
        """
        Use QFileDialog to open a file selection dialog and get the selected file path.
        """
        file_path, extention = QtWidgets.QFileDialog.getOpenFileName(self, "Select a right image", self.folder_path, "Image files (*.bmp)")
        print(file_path)
        pass

    def _find_corners(self, image):
        self.h, self.w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray, (11, 8))

        # refine the corners
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corner = cv2.cornerSubPix(gray, corner, winSize, zeroZone, criteria) 

        self.corners.append(corner)
        return ret, corner

    def find_corners_button_clicked(self):
        """
        For each image in the folder, find the corners of the chessboard in the image.
        """
        if self.folder_path is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a folder first.")
            return
        
        # create a window to show the images
        cv2.namedWindow("show_image", cv2.WINDOW_NORMAL)

        # read every .bmp file in the folder
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith(".bmp"):
                file_path = os.path.join(self.folder_path, file_name)
                print(file_path)
                image = cv2.imread(file_path)

                ret, corner = self._find_corners(image)

                show_image = cv2.drawChessboardCorners(image, (11, 8), corner, ret)
                show_image = cv2.resize(show_image, (0, 0), fx=0.5, fy=0.5)

                print("Number of corners: ", len(corner))
                
                # cv2.imshow("show_image", show_image)
                # wait for 1 second
                # cv2.waitKey(1000)
        
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_intrinsic_button_clicked(self):
        """
        Find the intrinsic parameters of the camera.
        """
        if self.folder_path is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a folder first.")
            return
        if len(self.corners) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please find the corners first.")
            return

        objpoints = np.zeros((1, 88, 3), np.float32)
        objpoints[0,:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # duplicate the points to match the number of corners
        objpoints = np.repeat(objpoints, len(self.corners), axis=0)
        
        # Initialize camera matrix
        camera_matrix = np.zeros((3, 3), np.float32)
        dist_coeffs = np.zeros((5, 1), np.float32)
        
        # Calibrate camera
        ret, intrinsic_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, self.corners, (self.w, self.h), 
            camera_matrix, dist_coeffs)
        
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.dist = dist

        print('\nIntrinsic:')
        print(intrinsic_matrix)

    def find_extrinsic_button_clicked(self):
        """
        Find the extrinsic parameters of the camera.
        """
        if len(self.corners) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please find the corners first.")
            return
        if self.rvecs is None or self.tvecs is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please find the intrinsic first.")
            return
        
        # From UI.find_extrinsic_spinBox get whitch number of image is selected
        image_index = self.ui.find_extrinsic_spinBox.value()
        if image_index < 0 or image_index >= len(self.corners):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a valid image index.")
            return
        
        # Get extrinsic matrix
        rotation_matrix = cv2.Rodrigues(self.rvecs[image_index])[0]
        extrinsic_matrix = np.hstack((rotation_matrix, self.tvecs[image_index]))
        print("Extrinsic:")
        print(extrinsic_matrix)

    def find_distortion_button_clicked(self):
        """
        Find the distortion parameters of the camera.
        """
        if self.dist is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please find the intrinsic first.")
            return
        
        print("Distortion:")
        print(self.dist)
        pass

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
        self.folder_path = 'C:/Users/sean/Desktop/cvdl_hw1/resources/Dataset_CvDl_Hw1/Q2_Image'
        self.corners = []
        self.h = None
        self.w = None
        self.rvecs = None
        self.tvecs = None
        self.dist = None
        self.inst = None
        self.index = None
    def setup_control(self):
        self.ui.load_folder_button.clicked.connect(self.load_folder_button_clicked)
        self.ui.load_image_l_button.clicked.connect(self.load_image_l_button_clicked)
        self.ui.load_image_r_button.clicked.connect(self.load_image_r_button_clicked)
        self.ui.find_corners_button.clicked.connect(self.find_corners_button_clicked)
        self.ui.find_intrinsic_button.clicked.connect(self.find_intrinsic_button_clicked)
        self.ui.find_extrinsic_button.clicked.connect(self.find_extrinsic_button_clicked)
        self.ui.find_distortion_button.clicked.connect(self.find_distortion_button_clicked)
        self.ui.show_result_button.clicked.connect(self.show_result_button_clicked)
        self.ui.show_words_on_borad_button.clicked.connect(self.show_words_on_borad_button_clicked)
        self.ui.show_words_vertical_button.clicked.connect(self.show_words_vertical_button_clicked)

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
        if self.folder_path is None or self.folder_path == "":
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a folder first.")
            return

        # filelist is a list of .bmp files in the folder
        filelist = [file for file in os.listdir(self.folder_path) if file.endswith(".bmp")]

        # reset the corners
        self.corners = []

        # read every .bmp file in the folder
        for index, file in enumerate(filelist):
            file_path = os.path.join(self.folder_path, file)
            print(file_path)
            image = cv2.imread(file_path)

            ret, corner = self._find_corners(image)

            show_image = cv2.drawChessboardCorners(image, (11, 8), corner, ret)
            show_image = cv2.resize(show_image, (0, 0), fx=0.25, fy=0.25)

            print("Number of corners: ", len(corner))
            
            cv2.imshow("show_image", show_image)
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
        ret, inst, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, self.corners, (self.w, self.h), 
            camera_matrix, dist_coeffs)
        
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.dist = dist
        self.inst = inst

        print('\nIntrinsic:')
        print(self.inst)

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
        self.index = self.ui.find_extrinsic_spinBox.value()
        if self.index <= 0 or self.index > len(self.corners):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a valid image index.")
            return
        
        # Get extrinsic matrix
        rotation_matrix = cv2.Rodrigues(self.rvecs[self.index-1])[0]
        extrinsic_matrix = np.hstack((rotation_matrix, self.tvecs[self.index-1]))
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

    def show_result_button_clicked(self):
        """
        Show the result of the camera calibration.
        """
        if self.dist is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please find the intrinsic first.")
            return
        if self.index is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a valid image index.")
            return

        # reload the image
        image_path = os.path.join(self.folder_path, f"{self.index}.bmp")
        image = cv2.imread(image_path)

        # get result image by cv2.undistort()
        result_image = cv2.undistort(image, self.inst, self.dist)

        # Create a black image for the titles
        title_height = 100
        combined_width = image.shape[1] * 2
        title_image = np.zeros((title_height, combined_width, 3), dtype=np.uint8)
        
        # Add titles to the image
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(title_image, "Distorted image", (image.shape[1]//4, 70), font, 2.5, (255,255,255), 2)
        cv2.putText(title_image, "Undistorted image", (image.shape[1]*5//4, 70), font, 2.5, (255,255,255), 2)
        
        # Combine images horizontally
        result_image = np.hstack((result_image, image))
        
        # Combine title and images vertically
        final_image = np.vstack((title_image, result_image))
        
        # Resize for display
        final_image = cv2.resize(final_image, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow("Camera Calibration Result", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _prepare_words_on_board(self):
        """
        Prepare the words on the board.
        """
        # get the words from UI.word_lineEdit
        words = self.ui.word_lineEdit.text()

        # if the words is empty, or the words is longer than 6 characters, or the words contains non-uppercase characters, return None
        if words == "" or len(words) > 6 or not words.isupper():
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a valid word.")
            return None
        
        # if self.inst, self.dist, self.rvecs, self.tvecs is None, return None
        if self.inst is None or self.dist is None or self.rvecs is None or self.tvecs is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please find the intrinsic first.")
            return None
        
        return words

    def show_words_on_borad_button_clicked(self):
        """
        Show the words on the board.
        """
        words = self._prepare_words_on_board()
        if words is None:
            return
        
        # Get the alphabet database
        fs = cv2.FileStorage(os.path.join(self.folder_path, "Q2_db", "alphabet_db_onboard.txt"), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            QtWidgets.QMessageBox.warning(self, "Warning", "Cannot open alphabet database file.")
            return
        
        # Get the points of all characters
        all_points = []
        all_segments = []
        segment_count = 0
        
        # The right down corner on the board of each character
        right_down_corner = [(7,5,0), (4,5,0), (1,5,0), (7,2,0), (4,2,0), (1,2,0)]

        for i, char in enumerate(words):
            # Get the points of the character from the database
            char_points = fs.getNode(char).mat()
            if char_points is None:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Cannot find points for character {char}")
                return
                
            # Ensure the points are float32 type and add the offset
            char_points = char_points.astype(np.float32)
            offset = np.array(right_down_corner[i], dtype=np.float32)
            
            # Process each segment
            for segment in char_points:
                # Add offset to both points in the segment
                start_point = segment[0] + offset
                end_point = segment[1] + offset
                
                all_points.append([start_point, end_point])
                all_segments.append((segment_count, segment_count + 1))
                segment_count += 2
        
        # Convert to the correct shape (N,1,3)
        objpoints = np.array(all_points, dtype=np.float32).reshape(-1, 1, 3)
        
        
        # Draw the projected points on all images in the folder
        for index in range(1, len(self.corners)+1):
            # Project the points to the image plane
            points_2d, _ = cv2.projectPoints(objpoints, 
                                            self.rvecs[index-1], 
                                            self.tvecs[index-1], 
                                            self.inst, 
                                            self.dist)
            
            image = cv2.imread(os.path.join(self.folder_path, f"{index}.bmp"))
            
            # Draw each segment
            for start_idx, end_idx in all_segments:
                pt1 = tuple(map(int, points_2d[start_idx][0]))
                pt2 = tuple(map(int, points_2d[end_idx][0]))
                cv2.line(image, pt1, pt2, (0, 0, 255), 20)

            # resize the image
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

            cv2.imshow("Words on Board", image)
            cv2.waitKey(1000)
        
        cv2.destroyAllWindows()

    def show_words_vertical_button_clicked(self):
        """
        Show the words vertically.
        """
        words = self._prepare_words_on_board()
        if words is None:
            return
        
        # Get the alphabet database
        fs = cv2.FileStorage(os.path.join(self.folder_path, "Q2_db", "alphabet_db_vertical.txt"), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            QtWidgets.QMessageBox.warning(self, "Warning", "Cannot open alphabet database file.")
            return
        
        # Get the points of all characters
        all_points = []
        all_segments = []
        segment_count = 0
        
        # The right down corner on the board of each character
        right_down_corner = [(7,5,0), (4,5,0), (1,5,0), (7,2,0), (4,2,0), (1,2,0)]
        
        for i, char in enumerate(words):
            # Get the points of the character from the database
            char_points = fs.getNode(char).mat()
            if char_points is None:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Cannot find points for character {char}")
                return
                
            # Ensure the points are float32 type and add the offset
            char_points = char_points.astype(np.float32)
            offset = np.array(right_down_corner[i], dtype=np.float32)
            
            # Process each segment
            for segment in char_points:
                # Add offset to both points in the segment
                start_point = segment[0] + offset
                end_point = segment[1] + offset
                
                all_points.append([start_point, end_point])
                all_segments.append((segment_count, segment_count + 1))
                segment_count += 2
        
        # Convert to the correct shape (N,1,3)
        objpoints = np.array(all_points, dtype=np.float32).reshape(-1, 1, 3)
        
        
        # Draw the projected points on all images in the folder
        for index in range(1, len(self.corners)+1):
            # Project the points to the image plane
            points_2d, _ = cv2.projectPoints(objpoints, 
                                            self.rvecs[index-1], 
                                            self.tvecs[index-1], 
                                            self.inst, 
                                            self.dist)
            
            image = cv2.imread(os.path.join(self.folder_path, f"{index}.bmp"))
            
            # Draw each segment
            for start_idx, end_idx in all_segments:
                pt1 = tuple(map(int, points_2d[start_idx][0]))
                pt2 = tuple(map(int, points_2d[end_idx][0]))
                cv2.line(image, pt1, pt2, (0, 0, 255), 20)

            # resize the image
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

            cv2.imshow("Words on Board", image)
            cv2.waitKey(1000)
        
        cv2.destroyAllWindows()
        
        pass

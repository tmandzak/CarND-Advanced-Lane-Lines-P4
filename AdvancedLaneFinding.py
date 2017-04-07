import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

class AdvancedLaneFinding:
    def __init__(self, cal_images, cal_nx, cal_ny):
        self.cal_images = cal_images
        self.cal_nx = cal_nx
        self.cal_ny = cal_ny
        self.findChessboardCorners()
        
        
    # Camera calibration
    def findChessboardCorners(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.cal_ny*self.cal_nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.cal_nx,0:self.cal_ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.cal_images)
        self.corners_images = []
        self.corners_images_failed = []

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.cal_nx, self.cal_ny), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                self.corners_images.append(cv2.drawChessboardCorners(img, (self.cal_nx, self.cal_ny), corners, ret))
            else:
                self.corners_images_failed.append(img)
                
    def _draw_images(self, images, titles=[], n=None, cols=2, show_axis='on'):
        if len(images)>0:
            if n or n==0:
                _ = plt.imshow(images[n])
            else:    
                rows = len(images) // cols + int(bool( len(images) % cols ))

                fig, axs = plt.subplots(rows, cols, figsize=(15, rows*4))
                axs = axs.ravel()

                i = 0
                for image in images:
                    a = axs[i] #axs[i // cols, i % cols]
                    a.axis(show_axis)
                    if len(titles)==len(images):
                        a.set_title(titles[i], fontsize=20)
                    a.imshow(image)
                    i+=1
 
    def draw_corners_images(self, n=None):
        self._draw_images(self.corners_images, n=n)
        
    def draw_corners_images_failed(self, n=None):
        self._draw_images(self.corners_images_failed, n=n)        
        
    # Do camera calibration given object points and image points    
    def calibrateCamera(self, img_size):
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)
        
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    # Test undistortion on an image
    def draw_test_undistort(self, test_image):
        img = cv2.imread(test_image)
        img_size = (img.shape[1], img.shape[0])
        self.calibrateCamera(img_size)
        dst = self.undistort(img)
        # Visualize undistortion
        self._draw_images(images=[img, dst], titles=['Original Image', 'Undistorted Image'])
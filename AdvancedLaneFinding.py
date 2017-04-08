import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class AdvancedLaneFinding:
    def __init__(self, cal_images, cal_nx, cal_ny, test_images):
        self.cal_images = [mpimg.imread(img) for img in glob.glob(cal_images)] 
        self.cal_nx = cal_nx
        self.cal_ny = cal_ny
        self.findChessboardCorners()
        
        self.test_images = [mpimg.imread(img) for img in glob.glob(test_images)]
        
        self.apex_h = 0.04
        self.apex_v = 0.59
        
        # Define a four sided polygon to mask
        self.imshape = self.test_images[0].shape
            
        imshape = self.imshape
        self.mask_vertices = np.array([[(0,imshape[0]),
                              (imshape[1]/2-imshape[1]*self.apex_h/2, imshape[0]*self.apex_v),
                              (imshape[1]/2+imshape[1]*self.apex_h/2, imshape[0]*self.apex_v),
                              (imshape[1], imshape[0])]], dtype=np.int32)
        
        
    # Camera calibration
    def findChessboardCorners(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.cal_ny*self.cal_nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.cal_nx,0:self.cal_ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        self.corners_images = []
        self.corners_images_failed = []

        # Step through the list and search for chessboard corners
        for img in self.cal_images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
        
    def undistort(self, img_input):
        if type(img_input) == list:
            img_otput = []
            for img in img_input:
                img_otput.append(cv2.undistort(img, self.mtx, self.dist, None, self.mtx))
        else:
            img_otput = cv2.undistort(img_input, self.mtx, self.dist, None, self.mtx)
                
        return img_otput
    
    # Test undistortion on an image
    def draw_test_undistort(self, test_image):
        img = cv2.imread(test_image)
        img_size = (img.shape[1], img.shape[0])
        self.calibrateCamera(img_size)
        dst = self.undistort(img)
        # Visualize undistortion
        self._draw_images(images=[img, dst], titles=['Original Image', 'Undistorted Image'])
        
    def _combinelists(self, l1, l2):
        res = []
        for i in range(len(l1)):
            res.append(l1[i])
            res.append(l2[i])
            
        return res    
        
    def draw_test_images_undistort(self):
        if len(self.test_images)>0:
            self.calibrateCamera(img_size = (self.test_images[0].shape[1], self.test_images[0].shape[0]) )

            dst_images = self.undistort(self.test_images)

            self._draw_images(images=self._combinelists(self.test_images, dst_images), titles=['Original Image', 'Undistorted Image']*len(dst_images))
            
            self.test_images = dst_images
            
    def mixed_threshold(self, img):
        # Color threshold
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  
        lower_yellow = np.array([20, 100, 100], dtype ='uint8')
        upper_yellow = np.array([30, 255, 255], dtype ='uint8')
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
        mask_white = cv2.inRange(img_gray, 200, 255)
        
        sbinary = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Threshold x gradient
        # Sobel x
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        sxbinary = cv2.inRange(scaled_sobel, 20, 100)
        
        mixed =  cv2.bitwise_or(sbinary, sxbinary)
        
        return mixed, sbinary, sxbinary        

    def draw_test_images_color_threshold(self):
        if len(self.test_images)>0:
            images = []
            for img in self.test_images:
                images.append(self.mixed_threshold(img)[1])
                
            self._draw_images(images=self._combinelists(self.test_images, images), titles=['Undistorted Image', 'Color thresholds']*len(images))
            
    def draw_test_images_gradient_threshold(self):
        if len(self.test_images)>0:
            images = []
            for img in self.test_images:
                images.append(self.mixed_threshold(img)[2])
                
            self._draw_images(images=self._combinelists(self.test_images, images), titles=['Undistorted Image', 'Gradient thresholds']*len(images))
            
            
    def draw_test_images_mixed_threshold(self):
        if len(self.test_images)>0:
            images = []
            for img in self.test_images:
                _, sbinary, sxbinary = self.mixed_threshold(img)
                images.append( np.dstack(( np.zeros_like(sxbinary), sxbinary, sbinary)) )
                
            self._draw_images(images=self._combinelists(self.test_images, images), titles=['Undistorted Image', 'Thresholded Binary']*len(images))        
        self.test_images = images    
            
        
    def region_of_interest(self, img):
        #defining a blank mask to start with
        mask = np.zeros_like(img)   

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, self.mask_vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
        
    def draw_test_images_masked(self):
        if len(self.test_images)>0:
            images = []
            for img in self.test_images:
                images.append(self.region_of_interest(img))
                
            self._draw_images(images=self._combinelists(self.test_images, images), titles=['Input', 'Masked']*len(images))    
        

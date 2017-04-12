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
        #self.processed_images = [img.copy() for img in self.test_images]
        imshape = self.test_images[0].shape
        
        # Define a four sided polygon to mask
        self.apex_h = 0.04
        self.apex_v = 0.59        
        self.mask_vertices = np.array([[(0,imshape[0]),
                              (imshape[1]/2-imshape[1]*self.apex_h/2, imshape[0]*self.apex_v),
                              (imshape[1]/2+imshape[1]*self.apex_h/2, imshape[0]*self.apex_v),
                              (imshape[1], imshape[0])]], dtype=np.int32)
        
        # Define four sided polygons for perspective transform
        #self.src_poly = np.float32([[160,720],[590,450],[705,450],[1280,720]])
        self.src_poly = np.float32([[0,720],[575,450],[705,450],[1280,720]])
        self.dst_poly = np.float32([self.src_poly[0],[self.src_poly[0][0],0],[self.src_poly[3][0],0],self.src_poly[3]])
        self.img_size = (imshape[1], imshape[0])
        self.M = cv2.getPerspectiveTransform(self.src_poly, self.dst_poly)
        
        self.src_poly_int = np.int32(self.src_poly).reshape((-1,1,2))
        self.dst_poly_int = np.int32(self.dst_poly).reshape((-1,1,2))
        
        
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
                
    def _draw_images(self, images, titles=[], n=None, cols=2, show_axis='on', cmap=None):
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
                    a.imshow(image, cmap)
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
        
    def draw_test_images_undistort(self, test_images = None):
        if test_images == None:
            test_images = self.test_images
            
        self.calibrateCamera(img_size = (test_images[0].shape[1], test_images[0].shape[0]) )
        dst_images = self.undistort(test_images)
        self._draw_images(images=self._combinelists(test_images, dst_images), titles=['Original Image', 'Undistorted Image']*len(dst_images))
            
        return dst_images
            
    def mixed_threshold(self, img):
        # Color threshold
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  
        lower_yellow = np.array([10, 100, 100], dtype ='uint8')
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

    def draw_test_images_color_threshold(self, test_images = None):
        if test_images == None:
            test_images = self.test_images
            
        images = []
        for img in test_images:
            images.append(self.mixed_threshold(img)[1])
                
        self._draw_images(images=self._combinelists(test_images, images), titles=['Undistorted Image', 'Color thresholds']*len(images))
        return images
            
            
    def draw_test_images_gradient_threshold(self, test_images = None):
        if test_images == None:
            test_images = self.test_images
            
        images = []
        for img in test_images:
            images.append(self.mixed_threshold(img)[2])

        self._draw_images(images=self._combinelists(test_images, images), titles=['Undistorted Image', 'Gradient thresholds']*len(images))
        return images
            
            
    def draw_test_images_mixed_threshold(self, test_images = None):
        if test_images == None:
            test_images = self.test_images
            
        images_color = []
        images_binary = []
        
        for img in test_images:
            mixed, sbinary, sxbinary = self.mixed_threshold(img)
            
            images_binary.append( mixed )
            images_color.append( np.dstack(( np.zeros_like(sxbinary), sxbinary, sbinary)) )
            
                
        self._draw_images(images=self._combinelists(test_images, images_color), titles=['Undistorted Image', 'Thresholded Binary']*len(images_color))        
        return images_binary, images_color   
            
       
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

    def warpPerspective(self, img):
        warped = cv2.warpPerspective(img, self.M, self.img_size)
        return warped
        
    def draw_test_images_warped(self, test_images = None):
        if test_images == None:
            test_images = self.test_images
            
        src_images = []
        dst_images = []
        ret_images = []
        
        for img in test_images:
            src_img = img.copy()
            cv2.polylines(src_img, [self.src_poly_int], True, (255,0,0), 5)
            src_images.append(src_img)            
            
            warped = self.warpPerspective(img)
            dst_img = warped.copy()
            cv2.polylines(dst_img, [self.dst_poly_int], True, (255,0,0), 5)
            dst_images.append(dst_img)

            ret_images.append(warped)

        self._draw_images(images=self._combinelists(src_images, dst_images), titles=['Input', 'Transformed']*len(src_images))     
        
        return ret_images

    def locateLaneLines(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
        #----------------------------
    
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    

        #----------------------------    

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)

        pts = np.int32(np.round(list(zip(left_fitx, ploty))))
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img=result,pts=[pts],isClosed=False,color=(255,255,0), lineType=8, thickness = 3)

        pts = np.int32(np.round(list(zip(right_fitx, ploty))))
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img=result,pts=[pts],isClosed=False,color=(255,255,0), lineType=8, thickness = 3)

        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)    

        return result
    
    def draw_binary_images_lanes_located(self, binary_images):
        images = []
        for img in binary_images:
            images.append(self.locateLaneLines(img))

        self._draw_images(images=self._combinelists(binary_images, images), titles=['Input', 'Output']*len(images))
        return images
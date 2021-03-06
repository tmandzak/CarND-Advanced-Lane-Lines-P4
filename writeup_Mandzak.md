# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_undistorted.png "Undistorted Chessboard"
[image2]: ./output_images/test_images_undistorted.png "Undistorted Test Images"
[image3]: ./output_images/test_images_thresholded.png "Thresholded test Images"
[image4]: ./output_images/test_images_perspective.png "Perspective transform on test images"
[image5]: ./output_images/test_images_thresholded_perspective.png "Perspective transform on thresholded images"
[image6]: ./output_images/test_images_lanes_located.png "Locating Lane Lines"
[image7]: ./output_images/test_images_final.png "Full pipeline applied to test images"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! The whole solution is implemented as a class called `AdvancedLaneFinding` defined in [AdvancedLaneFinding.py](https://github.com/tmandzak/CarND-Advanced-Lane-Lines-P4/blob/master/AdvancedLaneFinding.py).
The class is instanciated in [P4_Mandzak.ipynb](https://github.com/tmandzak/CarND-Advanced-Lane-Lines-P4/blob/master/P4_Mandzak.ipynb) in order to run and test the pipeline step by step and generate illustrations. Methods starting with `draw_` are responsible for making illustrations applying corresponding pipeline methods. Full run of `P4_Mandzak.ipynb` will produce all [images](https://github.com/tmandzak/CarND-Advanced-Lane-Lines-P4/tree/master/output_images) needed for this writeup as well as a [final video](https://github.com/tmandzak/CarND-Advanced-Lane-Lines-P4/blob/master/project_video_output.mp4). Further lines and cells references correspond to `AdvancedLaneFinding.py` and `P4_Mandzak.ipynb` respectively.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the constructor of `AdvancedLaneFinding` class in **lines 10-22** of the file called `AdvancedLaneFinding.py`. First three parameters of the constructor serve as input for calibration. `cal_images = 'camera_cal/calibration*.jpg'` defines a set of calibration images, `cal_nx = 9, cal_ny = 6` defines size of the chess board (see **cell 3** of the IPython notebook). **Line 14** loads `self.findChessboardCorners` method (**lines 44-72**) that initializes `self.objpoints` and `self.imgpoints` lists used then by `cv2.calibrateCamera` method (**line 22**) to compute the camera calibration and distortion coefficients `self.mtx` and `self.dest`. Undistort is implemented in `undistort` method in **lines 99-100**.
I applied this distortion correction to the test image in **cell 6** using the `draw_test_undistort` method (**lines 102-108**) and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I apply the distortion correction to the test images:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (`mixed_threshold` method at **lines 129-150** in `AdvancedLaneFinding.py` file).  Here's an example of my output for this step generated in the **cell 8** by `draw_test_images_mixed_threshold` method (**lines 176-191**). Blue and green colors represent results of color and gradient thresholding respectively:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a method called `warpPerspective`, which appears in lines **194-196**.  The `warpPerspective` method takes as inputs an image (`img`), as well as `self.M` matrix initialized once together with `self.Minv` matrix in the constructor code (**lines 28-29**). Source (`self.src_poly`) and destination (`self.dst_poly`) points used by `cv2.getPerspectiveTransform` method are hardcoded in **lines 25-26** as follows:  

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 191, 719      | 300, 719      | 
| 587, 454      | 300, 0        |
| 693, 454      | 1000, 0       |
| 1118, 719     | 1000, 719     |

I verified that my perspective transform was working as expected by drawing the `self.src_poly` and `self.dst_poly` points onto a test image and its warped counterpart (**cells 9-10**, `draw_test_images_warped` method defined in **lines 198-230**) to verify that the lines appear parallel in the warped image.

![alt text][image4]

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Code responsible for locating the lane lines and fitting them with 2nd order polynomials is placed in `locateLaneLines` method in **lines 232-354**. This method also returns lane lines curvatures `left_curverad, right_curverad, center_curverad` and car offset from the center of the lane `offset`. Ass suggested in the class lesson full histogram-based window search is performed only for initial recognition and after a fail (**lines 240-300**), for consequent frames faster search based on previous result is applied (**lines 302-310**). Offset is computed in **lines 333-335** and curvatures in **lines 338-348**.
To present results of lane lines detection following images where generated in **cell 11** with help of `draw_binary_images_lanes_located` method (**lines 401-415**):

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in **lines 333-348** of `locateLaneLines` method as described in a previous section.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the **cell 14** by calling `draw_test_images_pipeline` method (**lines 477-483**) based on a key `pipeline` method (**lines 469-475**) used also for processing frames during final video generation.
Here are examples of my result on test images:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project I've applied color thresholding that worked well for the challenging part of the Project 1. Applying only horizontal gradient thresholding let me get rid of horizontal distractions in a much efficient way than I've implemented in Project 1.
In order to make lane lines detection faster full window-search is performed only for the first run or after a failure, otherwise approach based on previous frame results is used.
In order to tackle challenges of other two advanced videos at least these steps still have to be implemented:
- implement more advanced outlier rejection that can filter out non-lane lines and noises
- implement adding each new detection to a weighted mean of the position of the lines to avoid jitter
- experiment more with color thresholding

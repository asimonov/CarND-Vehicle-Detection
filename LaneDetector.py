import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import matplotlib.image as mpimg

#Apply the distortion correction to the raw image.
# inputs:
#   img  -- input image
#   mtx  -- camera matrix from calibrateCamera
#   dist -- distortion coefficients from calibrateCamera
def undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# Apply Sobel along x or y axis to img.
# Then takes an absolute value and apply thresholds.
# Returns binary image.
#   inputs
#     img          -- input image
#     orient       -- 'x' or 'y'
#     kernel_size  -- sobel operator kernel size, in px
#     thresholds   -- two values for low and high thresholds
def sobel_abs_thresholds(img, orient='x', kernel_size=3, thresholds=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresholds[0]) & (scaled_sobel <= thresholds[1])] = 1
    return binary_output

# Applies Sobel along x and y.
# Computes the magnitude of the gradient and applies thresholds.
# Returns binary image.
#   inputs
#     img          -- input image
#     kernel_size  -- sobel operator kernel size, in px
#     thresholds   -- two values for low and high thresholds
def sobel_magnitude_thresholds(img, kernel_size=3, thresholds=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresholds[0]) & (scaled_sobel <= thresholds[1])] = 1
    return binary_output

# Applies Sobel along x and y, then computes the direction of the gradient for absolute sobel values.
# Applies thresholds.
# Returns binary image.
#   inputs
#     img          -- input image
#     kernel_size  -- sobel operator kernel size, in px
#     thresholds   -- two values for low and high thresholds
def sobel_graddir_thresholds(img, kernel_size=3, thresholds=(0., np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_grad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(sobel_grad).astype(np.uint8)
    binary_output[(sobel_grad >= thresholds[0]) & (sobel_grad <= thresholds[1])] = 1
    return binary_output

# Converts image to HLS color space.
# Applies thresholds to the S-channel.
# Returns binary image.
#   inputs
#     img          -- input image
#     thresholds   -- two values for low and high thresholds
def hls_s_thresholds(img, thresholds=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s >= thresholds[0]) & (s <= thresholds[1])] = 1
    return binary_output

# Produce binary image from original color image where lane lines are as distinct as possible for later detection
#
# plt input is pyplot object. If passed in the function shows the steps arriving at final binary image
#
# the final image can be described as
#
#    binary = (sx & smag) || ( (sy & sgrad) || ( (smag & sgrad) || (sx || s) ) ) 
#
# where 
#    sx is sobelx, 
#    smag is sobel gradient magnitude, 
#    sy is sobely, 
#    sgrad is sobel gradient,
#    s is S channel in HLS space
def thresholded_binary(image, plt=None):
    # sobel x
    sx_kernel = 3
    sx_thresh = (20, 100)
    sx_binary = sobel_abs_thresholds(image, orient='x', kernel_size=sx_kernel, thresholds=sx_thresh)
    # sobel y
    sy_kernel = 3
    sy_thresh = (30, 100)
    sy_binary = sobel_abs_thresholds(image, orient='y', kernel_size=sy_kernel, thresholds=sy_thresh)
    # sobel magnitude
    smag_kernel = 5
    smag_thresh = (30,100)
    smag_binary = sobel_magnitude_thresholds(image, kernel_size=smag_kernel, thresholds=smag_thresh)
    # sobel gradient direction
    sgraddir_kernel = 7
    sgraddir_thresh = (0.7, 1.3)
    sgraddir_binary = sobel_graddir_thresholds(image, kernel_size=sgraddir_kernel, thresholds=sgraddir_thresh)
    # HSV's S threshold
    s_thresh = (170,255)
    s_binary = hls_s_thresholds(image, thresholds=s_thresh)
    # x gradient and S threshold
    combined = np.zeros_like(sx_binary)
    combined[(s_binary == 1) | (sx_binary == 1)] = 1
    # x gradient and magnitude
    combined0 = np.zeros_like(sx_binary)
    combined0[(sx_binary == 1) & (smag_binary == 1)] = 1   
    # y gradient and grad direction
    combined3 = np.zeros_like(sy_binary)
    combined3[(sy_binary == 1) & (sgraddir_binary == 1)] = 1    
    # use previous combined, but add pixels where gradient magnitude and direction are activated
    combined2 = np.copy(combined)
    combined2[((smag_binary == 1) & (sgraddir_binary == 1))] = 1
    # combined2 OR combined3
    combined4 = np.copy(combined2)
    combined4[combined3 == 1] = 1
    # combined4 OR combined0
    combined5 = np.copy(combined4)
    combined5[combined0 == 1] = 1
    # plot intermediate output
    if plt is not None:
        n = 11
        f, axes = plt.subplots(n, 1, figsize=(20,10*n));
        axes[0].imshow(sx_binary, cmap='gray');
        axes[0].set_title('sobel x (sx)', fontsize=30);
        axes[1].imshow(sy_binary, cmap='gray');
        axes[1].set_title('sobel y (sy)', fontsize=30);
        axes[2].imshow(smag_binary, cmap='gray');
        axes[2].set_title('sobel grad magnitude (smag)', fontsize=30);
        axes[3].imshow(sgraddir_binary, cmap='gray');
        axes[3].set_title('sobel grad direction (sgrad)', fontsize=30);        
        axes[4].imshow(s_binary, cmap='gray');
        axes[4].set_title('HLS S-channel (s)', fontsize=30);
        
        axes[5].imshow(combined, cmap='gray');
        axes[5].set_title('sx OR s', fontsize=30);
        axes[6].imshow(combined0, cmap='gray');
        axes[6].set_title('sx AND smag', fontsize=30);
        axes[7].imshow(combined3, cmap='gray');
        axes[7].set_title('sy AND sgrad', fontsize=30);
        axes[8].imshow(combined2, cmap='gray');
        axes[8].set_title('(sx OR s) OR (smag AND sgrad)', fontsize=30);
        axes[9].imshow(combined4, cmap='gray');
        axes[9].set_title('(sy AND sgrad) OR    [ (sx OR s) OR (smag AND sgrad) ]', fontsize=30);
        axes[10].imshow(combined5, cmap='gray');
        axes[10].set_title('(sx AND smag) AND { (sy AND sgrad) OR    [ (sx OR s) OR (smag AND sgrad) ]  }', fontsize=30);
    return combined5

# Does perspective transform for the part of the road between lanes.
# Takes undistorted image
# Returns transfromed image, the matrix used to transform it and the matrix for inverse transform
def perspective_transform(img, M=None, Minv=None):
    imshape = img.shape
    
    if (M is None):
        src = np.float32([[85,670], 
                          [515,480], 
                          [765,480], 
                          [1195,670]])
        dst = np.float32([[imshape[1]*.0, imshape[0]*1.],
                          [imshape[1]*.0, imshape[0]*.0],
                          [imshape[1]*1., imshape[0]*.0],
                          [imshape[1]*1., imshape[0]*1.]])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# Define a class to receive the characteristics of each line detection
class Line():
    # class constants that are same for all instances
    
    # pct width of total image that one lane can occupy. used to decide on histograms
    _pct_image_width_lane_width_max=0.1
    # for the sliding window method -- height of the sliding window
    _pct_image_height_slice=0.1
    # Define conversions in x and y from pixels space to meters
    _ym_per_pix = 3/130 # meters per pixel in y dimension. based on dashed line=3m
    _xm_per_pix = 3.7/840 # meteres per pixel in x dimension
    # history length for confirming detection
    _n = 5
    
    
    def __init__(self, n=5):
        # making this instance variable. it takes precedence over class variable.
        # used to create an instance for individual frame detection (n=1)
        self._n = n
        # was the line detected in the last iteration?
        self._detected = False
        # number of last recent failed fits
        self._failed_fits = 0
        # x values of the last n fits of the line
        self._recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self._bestx = None
        #polynomial coefficients averaged over the last n iterations
        self._best_fit = None
        #polynomial coefficients for the most recent fit in px coordinates
        self._current_fit_px = [np.array([False])]  
        #polynomial coefficients for the most recent fit in m coordinates
        self._current_fit_m = [np.array([False])]  
        #radius of curvature of the line in pixels
        self._radius_of_curvature = None 
        #radius of curvature of the line in meters
        self._radius_of_curvature_m = None 
        #distance in px of vehicle center from the line
        self._line_pos_px = None 
        #distance in m of vehicle center from the line
        self._line_pos_m = None 
        #difference in fit coefficients between last and new fits
        self._diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line image pixels
        self._img_allx = None  
        #y values for detected line image pixels
        self._img_ally = None
        #x values for detected sliding window pixels
        self._histogram_allx = None  
        #y values for detected sliding pixels
        self._histogram_ally = None
        #x values for fitted pixels
        self._line_allx = None  
        #y values for fitted pixels
        self._line_ally = None
        #devirative of fitted line wrt y at the bottom of top-down view, in px coordinates
        self._y_deriv = None
        
    # find radius of curvature closest to the bottom of the image
    def calc_radius_of_curvature(self):
        # curvature in pixels
        a = self._current_fit_px[0]
        b = self._current_fit_px[1]
        c = self._current_fit_px[2]
        y_eval = np.max(self._line_ally)
        self._radius_of_curvature = (1.0+(2.0*a*y_eval+b)**2)**1.5 / np.abs(2*a)
        # curvature in meters
        self._current_fit_m = np.polyfit(self._line_ally*self._ym_per_pix, self._line_allx*self._xm_per_pix, 2)
        a = self._current_fit_m[0]
        b = self._current_fit_m[1]
        c = self._current_fit_m[2]
        self._radius_of_curvature_m = ((1.0 + (2.0*a*y_eval + b)**2)**1.5) / np.abs(2*a)
        # derivative of line wrt y at the bottom
        self._y_deriv = 2*a*y_eval + b

        
    # given binary top-down view of lane pixels find possible left and right x values, in pixels
    # using histogram method
    # Used as static method. Does not change Line state.
    #    inputs:
    #       image                           -- binary image with top-down (perspective-transformed) view of lane lines. 
    #                                       -- assume x and y coordinates are reversed
    #    outputs:
    #        left_lane_x_px                 -- initial estimate for left lane x coordinate or None
    #        right_lane_x_px                -- initial estimate for right lane x coordinate or None
    def find_left_right_x(self, image):
        histogram1 = np.sum(image, axis=0)
        width = len(histogram1)
        # find the most prominent peak in the histogram
        imax1 = np.argmax(histogram1)
        max1 = histogram1[imax1]
        # remove the peak and pct_image_width_lane_width_max elements around it.
        histogram2 = np.copy(histogram1)
        nx = width*self._pct_image_width_lane_width_max
        histogram2[imax1-int(nx/2.0) : imax1+int(nx/2.0)] = 0
        # find second highest peak
        imax2 = np.argmax(histogram2)
        max2 = histogram2[imax2]
        # remove the second peak and pct_image_width_lane_width_max elements around it.
        histogram3 = np.copy(histogram2)
        histogram3[imax2-int(nx/2.0) : imax2+int(nx/2.0)] = 0
        # find third highest peak
        imax3 = np.argmax(histogram3)
        max3 = histogram3[imax3]

        l = min(imax1, imax2)
        r = max(imax1, imax2)

        # decide what kind of situation we have
        if max1/max3<1.5:
            # inconsequential difference between topmost peak and the 'background'
            return None, None
        if max2/max3<1.5:
            # inconsequential difference between second peak and the 'background'
            return None if imax2==l else l, None if imax2==r else r
        return l, r

    # use sliding window to find the potential line pixels.
    # then fit second order polynomial to represent the line
    #    inputs:
    #       image                           -- binary image with top-down (perspective-transformed) view of lane lines. 
    #                                       -- assume x and y coordinates are reversed
    #       initial_x                       -- x from which to start finding the line
    def fit_from_x_on_image(self, image, initial_x):
        (height, width) = image.shape # in the image operations x and y will be reversed
        # sliding window points
        nx = int(width*self._pct_image_width_lane_width_max)
        ny = int(height*self._pct_image_height_slice)
        y_points = np.arange(ny,height+1,ny)
        x_points = np.zeros_like(y_points)
        x_points[-1] = initial_x
        imgcopy = np.zeros_like(image)
        # sliding window loop
        failed_window_points = 0
        for i in range(len(y_points)-1,0,-1):
            slice_y_bottom = y_points[i]
            slice_y_top = slice_y_bottom - ny
            slice_x_bottom = x_points[i]
            imgslice = image[slice_y_top:slice_y_bottom,:]
            ones = np.zeros_like(imgslice)
            ones[:,slice_x_bottom-int(nx/2):slice_x_bottom+int(nx/2)] = 1
            imgslice2 = np.bitwise_and(imgslice, ones)
            imgcopy[slice_y_top:slice_y_bottom,:] = imgslice2
            (y,x) = np.nonzero(imgslice2) # x and y are here in the 'opposite' order
            if (len(set(y))>int(ny*.30)):
                # looks like 'vertical' line segment
                slice_x_top = np.mean(x)
            else:
                failed_window_points += 1
                if self._detected:
                    # if we fitted line successfully previously, use last fit for best estimate of top x in window
                    slice_x_top = np.polyval(self._current_fit_px, slice_y_top)
                else:
                    slice_x_top = slice_x_bottom
            x_points[i-1] = slice_x_top
        # fit polymonial to x,y found in sliding window loop
        last = self._current_fit_px
        self._current_fit_px = np.polyfit(y_points, x_points, 2)
        self._diff = self._current_fit_px - last
        self._line_ally = np.array(range(0,height))
        self._line_allx = np.polyval(self._current_fit_px, self._line_ally)
        # save/calculate additional data and y derivative at bottom (to test for parallel lines outside)
        self._img_ally, self._img_allx = np.nonzero(imgcopy)
        self.calc_radius_of_curvature() # also does y_deriv
        self._line_pos_px = -(int(width/2) - self._line_allx[-1]) #distance in px of the line vs vehicle center
        self._line_pos_m = self._line_pos_px * self._xm_per_pix   #distance in m of the line vs vehicle center 
        # remember sliding window points
        self._histogram_allx = x_points 
        self._histogram_ally = y_points
        
        # estimate confidence by looking at number of sliding window points that detect something like a line segment
        # or whether line derivative wrt y is too big
        if (failed_window_points > len(y_points)*0.70 or abs(self._y_deriv)>10.0 ):
            self._failed_fits += 1
        else:
            self._failed_fits = 0

        # draw extra annotations on temp result
        imgcopy = cv2.cvtColor(imgcopy*255, cv2.COLOR_GRAY2BGR)
        # plot points estimated from sliding histogram window
        for p in zip(self._histogram_allx, self._histogram_ally):
            cv2.circle(imgcopy, p, radius=6, color=(255,0,0), thickness=-1)
        # plot fitted lines
        cv2.polylines(imgcopy, 
                      np.int32([np.dstack([self._line_allx, self._line_ally])[0]]), 
                      isClosed=0, 
                      color=(0,0,255), thickness=5)
        # update historical lists
        self._recent_xfitted.append(self._line_allx[-1]) # bottom x from the fitted line
        if len(self._recent_xfitted)>self._n:
            self._recent_xfitted.pop(0)
        self._bestx = np.mean(self._recent_xfitted) # average bottom x from last few fits
        # average coefficients of the fit over last few fits
        i = len(self._recent_xfitted)
        if i==1:
            self._best_fit = np.zeros_like(self._current_fit_px)
        self._best_fit = (self._best_fit*(i-1) + self._current_fit_px) / float(i)
        if i==self._n:
            self._detected = True

        return imgcopy
    
    
    # go back to last good fit, but use it on image provided
    #    inputs:
    #       image                           -- binary image with top-down (perspective-transformed) view of lane lines. 
    #                                       -- assume x and y coordinates are reversed
    def use_last_good_fit(self, image):
        (height, width) = image.shape # in the image operations x and y will be reversed
        
        if len(self._recent_xfitted)<self._n:
            # not enough history
            return None
        
        # use last good coefficients
        self._current_fit_px = self._best_fit

        # sliding window points
        nx = int(width*self._pct_image_width_lane_width_max)
        ny = int(height*self._pct_image_height_slice)
        y_points = np.arange(ny,height+1,ny)
        x_points = np.zeros_like(y_points)
        x_points[-1] = self._bestx
        imgcopy = np.zeros_like(image)
        # sliding window loop
        for i in range(len(y_points)-1,0,-1):
            slice_y_bottom = y_points[i]
            slice_y_top = slice_y_bottom - ny
            slice_x_bottom = x_points[i]
            imgslice = image[slice_y_top:slice_y_bottom,:]
            ones = np.zeros_like(imgslice)
            ones[:,slice_x_bottom-int(nx/2):slice_x_bottom+int(nx/2)] = 1
            imgslice2 = np.bitwise_and(imgslice, ones)
            imgcopy[slice_y_top:slice_y_bottom,:] = imgslice2
            (y,x) = np.nonzero(imgslice2) # x and y are here in the 'opposite' order
            slice_x_top = np.mean(x)
            slice_x_top = np.polyval(self._current_fit_px, slice_y_top)
            x_points[i-1] = slice_x_top
        
        self._line_ally = np.array(range(0,height))
        self._line_allx = np.polyval(self._current_fit_px, self._line_ally)
        
        # save/calculate additional data and y derivative at bottom (to test for parallel lines outside)
        self._img_ally, self._img_allx = np.nonzero(imgcopy)
        self.calc_radius_of_curvature() # also does y_deriv
        self._line_pos_px = -(int(width/2) - self._line_allx[-1]) #distance in px of the line vs vehicle center
        self._line_pos_m = self._line_pos_px * self._xm_per_pix   #distance in m of the line vs vehicle center 
        # remember sliding window points
        self._histogram_allx = x_points 
        self._histogram_ally = y_points
        
        # just marking it to the outside as not 'great' fit
        self._failed_fits += 1

        # draw extra annotations on temp result
        imgcopy = cv2.cvtColor(imgcopy*255, cv2.COLOR_GRAY2BGR)
        # plot points estimated from sliding histogram window
        for p in zip(self._histogram_allx, self._histogram_ally):
            cv2.circle(imgcopy, p, radius=6, color=(255,0,0), thickness=-1)
        # plot fitted lines
        cv2.polylines(imgcopy, 
                      np.int32([np.dstack([self._line_allx, self._line_ally])[0]]), 
                      isClosed=0, 
                      color=(0,0,255), thickness=5)

        return imgcopy


# class to combine the lane detection pipeline and keep track of both lines and produce output images
class LaneDetector:
    # initial number of images to be processed before detecting lane, assuming video
    # can be overwrittent in instance to deal with individual frame detection
    _initial_images_number = 5
    # maximum difference of derivatives between two lines at the bottom
    _max_deriv_diff = 5.0
    # maximum lane width in meters
    _max_lane_width_m = 4.4
    # miminum lane width in meters
    _min_lane_width_m = 1.7
        
    # takes distortion matrix and coefficients calibrated for the camera
    # also takes initial_images_number to overwrite
    def __init__(self, mtx, dist, initial_images_number=5):#LaneDetector._initial_images_number):
        # making this instance variable. it takes precedence over class variable
        self._initial_images_number = initial_images_number
        # undistortion matrix and coefficients
        self._mtx = np.copy(mtx)
        self._dist = np.copy(dist)
        # objects to track left and right lines
        self._lline = Line(n=self._initial_images_number)
        self._rline = Line(n=self._initial_images_number)
        # original image
        self._original_image = None
        # un-distorted image
        self._undistorted_image = None
        # thresholded binary combining different edge/contrast detection techniques
        self._thresholded_binary_image = None
        # road in front, perspective-transformed
        self._top_down_binary_image = None
        # perspective transform matrix and inverse
        self._M, self._Minv = None, None
        # binary image of road ahead with annotated fitted lines
        self._top_down_binary_with_lines_image = None
        # original undistorted image annotated with fitted lines
        self._original_annotated_image = None
        # image counter
        self._image_number = 0
        # number of bad sequential frames we encountered
        self._bad_frames = 0
        self._last_bad_frame = 0
        
    # put text on top-down binary view
    def annotate_top_down_binary(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # curvature
        if self._lline._detected:
            lc = '{:,.2f}'.format(self._lline._radius_of_curvature)
        else:
            lc = 'NAN'
        if self._rline._detected:
            rc = '{:,.2f}'.format(self._rline._radius_of_curvature)
        else:
            rc = 'NAN'
        s = 'curvature: Left {} px, Right {} px'.format(lc, rc)
        cv2.putText(self._top_down_binary_with_lines_image, s, (350,100), font, fontScale=0.7, color=(0,255,0))
        # line distance from car center
        if self._lline._detected:
            ld = '{:,.2f}'.format(self._lline._line_pos_px)
        else:
            ld = 'NAN'
        if self._rline._detected:
            rd = '{:,.2f}'.format(self._rline._line_pos_px)
        else:
            rd = 'NAN'
        s2 = 'lane dist from center: Left {} px, Right {} px'.format(ld, rd)
        cv2.putText(self._top_down_binary_with_lines_image, s2, (350,150), font, fontScale=0.7, color=(0,255,0))
        s2 = 'derivs: Left {} px, Right {} px'.format(self._lline._y_deriv, self._rline._y_deriv)
        cv2.putText(self._top_down_binary_with_lines_image, s2, (350,200), font, fontScale=0.7, color=(0,255,0))

    # apply the lines and info text to the original undistorted image
    def annotate_undistorted_image(self):
        # draw area if both line detected
        _color_warp = np.zeros_like(self._original_image)
        if self._lline._detected and self._rline._detected:
            # Recast the x and y points into usable format for cv2.fillPoly()
            _pts_left = np.array([np.transpose(np.vstack([self._lline._line_allx, self._lline._line_ally]))])
            _pts_right = np.array([np.flipud(np.transpose(np.vstack([self._rline._line_allx, self._rline._line_ally])))])
            _pts = np.hstack((_pts_left, _pts_right))
            # Draw the lane onto the warped blank image
            cv2.fillPoly(_color_warp, np.int_([_pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        (height, width, _) = self._original_image.shape
        _newwarp = cv2.warpPerspective(_color_warp, self._Minv, (width,height)) 
        # Combine the result with the original image
        self._original_annotated_image = cv2.addWeighted(self._undistorted_image, 1, _newwarp, 0.3, 0)

        # draw left line if detected
        _color_warp = np.zeros_like(self._original_image)
        if (self._lline._detected):
            if (self._lline._failed_fits > 0):
                color = (255,0,0)
            else:
                color = (0,128,128)
            cv2.polylines(_color_warp, 
                          np.int32([np.dstack([self._lline._line_allx, self._lline._line_ally])[0]]), 
                          isClosed=0, 
                          color=color, thickness=60)
        # draw right line if detected
        if (self._rline._detected):
            if (self._rline._failed_fits > 0):
                color = (255,0,0)
            else:
                color = (0,128,128)
            cv2.polylines(_color_warp, 
                          np.int32([np.dstack([self._rline._line_allx, self._rline._line_ally])[0]]), 
                          isClosed=0, 
                          color=color, thickness=60)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        _newwarp = cv2.warpPerspective(_color_warp, self._Minv, (width,height)) 
        # Combine the result with the original image
        self._original_annotated_image = cv2.addWeighted(self._original_annotated_image, 1, _newwarp, 1.0, 0)
        
        
        # add text details
        _overlay = np.zeros_like(self._original_image)
        cv2.rectangle(_overlay, (250,60), (1100,200), (255,255,255), -1)
        self._original_annotated_image = cv2.addWeighted(self._original_annotated_image, 1, _overlay, 0.8, 0)
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        # curvature
        if self._lline._detected:
            if abs(self._lline._radius_of_curvature_m) < 10000:
                lc = '{:,.0f}m'.format(self._lline._radius_of_curvature_m)
            else:
                lc = 'STRAIGHT'
        else:
            lc = 'NAN'
        if self._rline._detected:
            if abs(self._rline._radius_of_curvature_m) < 10000:
                rc = '{:,.0f}m'.format(self._rline._radius_of_curvature_m)
            else:
                rc = 'STRAIGHT'
        else:
            rc = 'NAN'
        s = 'curvature: Left {}, Right {}'.format(lc, rc)
        cv2.putText(self._original_annotated_image, s, (300,100), 
                    font, fontScale=0.7, color=(0,0,0), lineType=cv2.LINE_AA)
        
        # line distance from car center
        if self._lline._detected:
            ld = '{:,.2f}m'.format(self._lline._line_pos_m)
        else:
            ld = 'NAN'
        if self._rline._detected:
            rd = '{:,.2f}m'.format(self._rline._line_pos_m)
        else:
            rd = 'NAN'
        s2 = 'lane distance from car center: Left {}, Right {}'.format(ld, rd)
        cv2.putText(self._original_annotated_image, s2, (300,130), 
                    font, fontScale=0.7, color=(0,0,0), lineType=cv2.LINE_AA)

        if self._lline._detected and self._rline._detected:
            s3 = 'car off center by: {:,.2f}m'.format(self._lline._line_pos_m+self._rline._line_pos_m)
        else:
            s3 = 'car off center by: UNDEFINED'
        cv2.putText(self._original_annotated_image, s3, (300,160), 
                    font, fontScale=0.7, color=(0,0,0), lineType=cv2.LINE_AA)
        
    # create 'diagnostic' view combining undistorted image with all final annotations
    # plus intermediate images used in the pipeline
    def create_diagnostic_view(self):
        (height, width, _) = self._original_image.shape
        self._diagnostic_view = np.zeros((height*2, width, 3), dtype=np.uint8)
        self._diagnostic_view[0:height, 0:width] = self._original_annotated_image
        rh = int(height/2)
        rw = int(width/2)
        # distorted original image
        self._diagnostic_view[height:(height+rh), 0:(rw)] = cv2.resize(self._original_image, (rw,rh), interpolation=cv2.INTER_AREA) 
        # thresholded binary
        img = cv2.cvtColor(self._thresholded_binary_image*255, cv2.COLOR_GRAY2BGR)
        self._diagnostic_view[height:(height+rh), rw:(2*rw)] = cv2.resize(img, (rw,rh), interpolation=cv2.INTER_AREA) 
        # perspective-transformed binary
        img = cv2.cvtColor(self._top_down_binary_image*255, cv2.COLOR_GRAY2BGR)
        self._diagnostic_view[(height+rh):(height+2*rh), 0:(rw)] = cv2.resize(img, (rw,rh), interpolation=cv2.INTER_AREA) 
        # annotated binary
        self._diagnostic_view[(height+rh):(height+2*rh), rw:(2*rw)] = cv2.resize(self._top_down_binary_with_lines_image, (rw,rh), interpolation=cv2.INTER_AREA) 
        
    # main method that takes images from stream of video, undistorts them, applies individual image pipeline
    # to create thresholded binary view of the road in front,
    # detects lines, checks if the detection is valid
    # then creates final annotated view, projecting detected lines and lane polygon on the original undistorted image
    def process_image(self, image, initial_images_number=1, return_diagnostic_views=False):
        self._original_image = np.copy(image)
        self._image_number += 1
        (height, width, _) = self._original_image.shape
        # un-distort image
        self._undistorted_image = undistort(self._original_image, self._mtx, self._dist)
        # produce thresholded binary
        self._thresholded_binary_image = thresholded_binary(self._undistorted_image)
        # perspective transform of the road in front
        self._top_down_binary_image, self._M, self._Minv = perspective_transform(self._thresholded_binary_image, 
                                                                                 self._M, self._Minv)

        # initial detection or detection after unsuccessful fits in preceding frames
        lx, rx = None, None
        if (self._image_number < self._initial_images_number or self._bad_frames > 5):
            # only use bottom half of the image for initial x detection using histogram method
            # build histogram of lower half of the image, closer to the car -- lines should be more straight
            lx, rx = Line().find_left_right_x(self._top_down_binary_image[int(height/2):,:])
            self._bad_frames = 0
        else:
            _rx = None
            if (not self._lline._detected or self._lline._failed_fits>0):
                # problems with left line detection in previous iterations. let's use histogram method to find it afresh
                lx, _rx = Line().find_left_right_x(self._top_down_binary_image[int(height/2):,:])
            if (not self._rline._detected or self._rline._failed_fits>0):
                # problems with right line detection in previous iterations. let's use histogram method to find it afresh
                if _rx is not None:
                    rx = _rx
                else:
                    lx_, rx = Line().find_left_right_x(self._top_down_binary_image[int(height/2):,:])
        # if initial detections or trying to detect the lines after unseccessful attempts
        # did not result in some estimate of x for left or right lines, use the best x estimates found so far
        if lx is None and self._lline._detected:
            lx = self._lline._bestx
        if rx is None and self._rline._detected:
            rx = self._rline._bestx
            
        # fit the lines from best x we found so far
        if lx is not None:
            l_imgcopy = self._lline.fit_from_x_on_image(self._top_down_binary_image, lx)
        if rx is not None:
            r_imgcopy = self._rline.fit_from_x_on_image(self._top_down_binary_image, rx)

        # test if detected lines are parallel
        if self._lline._detected and self._rline._detected \
                and abs(self._lline._y_deriv - self._rline._y_deriv) > self._max_deriv_diff:
            # high chance of wrong fit of either line or both
            print('frame {}: NOT PARALLEL'.format(self._image_number))
            if self._last_bad_frame+1 == self._image_number:
                self._bad_frames += 1
                self._last_bad_frame = self._image_number
            else:
                self._last_bad_frame = 0
                self._bad_frames = 0
            l_imgcopy = self._lline.use_last_good_fit(self._top_down_binary_image)
            r_imgcopy = self._rline.use_last_good_fit(self._top_down_binary_image)

        # test if lines are about the right distance, left is left and right is right
        if self._lline._detected and self._rline._detected \
                and (abs(self._rline._line_pos_m - self._lline._line_pos_m) > self._max_lane_width_m
                     or abs(self._rline._line_pos_m - self._lline._line_pos_m) < self._min_lane_width_m
                     or self._rline._line_pos_m < self._lline._line_pos_m
                    ):
            # high chance of wrong fit of either line or both
            print('frame {}: WRONG DISTANCE {}, {}, {}, {}'.format(self._image_number, lx, rx, self._lline._line_pos_m, self._rline._line_pos_m))
            if self._last_bad_frame+1 == self._image_number:
                self._bad_frames += 1
                self._last_bad_frame = self._image_number
            else:
                self._last_bad_frame = 0
                self._bad_frames = 0
            l_imgcopy = self._lline.use_last_good_fit(self._top_down_binary_image)
            r_imgcopy = self._rline.use_last_good_fit(self._top_down_binary_image)

        # combine both lines info in the top-down image
        if self._lline._detected and self._rline._detected:
            self._top_down_binary_with_lines_image = np.bitwise_or(l_imgcopy, r_imgcopy)
        elif not self._lline._detected and self._rline._detected:
            self._top_down_binary_with_lines_image = r_imgcopy
        elif self._lline._detected and not self._rline._detected:
            self._top_down_binary_with_lines_image = l_imgcopy
        else:
            self._top_down_binary_with_lines_image = cv2.cvtColor(self._top_down_binary_image*255, cv2.COLOR_GRAY2BGR)
        
        self.annotate_top_down_binary()

        self.annotate_undistorted_image()
                
        if (not return_diagnostic_views):
            return self._original_annotated_image
        else:
            self.create_diagnostic_view()
            return self._diagnostic_view







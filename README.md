## Udacity Self-Driving Car Engineer Nanodegree Project 5: Vehicle Detection

---

[//]: # (References)
[image1]: ./examples/hot_windows_example.png
[image2]: ./examples/example_diagnostics1.png
[image3]: ./examples/example_diagnostics2.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

The whole project is implemented in one Jupyter Notebook [`vehicle-detection.ipynb`](./vehicle-detection.ipynb). The notebook is well commented throughout and should be read as part of this documentation.

I first load data into `allimages` and `alllabels` variables.
In total we have 17760 images of cars and non-cars, almost equally split between these two classes.

I then define functions `features_spatial` (just pixel values of 32x32 resized image presented as one vector), `features_color_hist` (color histogram in specified color space, presented as one vector) and `features_hog` (HOG features extracted using skimage `hog` function and returned as one vector).
The notebook visualizes the features for couple of example images.

I then define `extract_features` that combines the three types of features above for an input image.

`extract_features` is run on the provided dataset with YUV colorspace and then sklearn `StandardScaler()` is calibrated on the full feature set.

I then train `sklearn.LinearSVC` classifier on the full labelled scaled feature set. It gives accuracy of 99.3% on random test set of 20% of the original data.

I then use `slide_window` function from the lectures to generate a set of windows covering ROI (region of interest), which roughly corresponds to horizontal area of input image from just above the bonnet to just over the horizon. I generate windows in two scales: 128x128 px and 64x64 px. They empirically produce the best results.

I then define function `search_windows` that takes full image and list of windows to search and runs classifier on those areas. It returns list of windows classified to be _cars_, as well as `decision_values` from SVM classifier (that correspond to distance of the image from the decision boundary, the bigger the value the more probable that this detection is `true positive`. The following image shows example hot windows identified, with red circle size proportional to decision function values:

![matched windows with probabilities as circles][image1]

`make_heatmap` and `apply_threshold` functions produce heatmap of the detections, taking into account decision function values. `draw_labeled_bboxes` function finds bounding boxes for identified blobs in heatmap, filters out what would be an improbable box for a car (too small in either direction, or too 'vertical').

`VehicleDetector` class defined to keep Vehicle Detection Pipeline parameters, as well as heatmaps in consecutive frames for averaging over video stream. `process_image` is the main working function in this class, which combines calls to functions mentioned above plus the tracking code that averages the heatmaps and produces final detections. `process_diagnostic_views` function is a wrapper around `process_image` that adds extra information to the final image, detailing steps of the pipeline like window detections, heatmaps etc. Its output is demonstrated next (sorry about the non-matching scales as its not easy to grab the whole image on one screen):

![diagnostic view 1][image2]
![diagnostic view 2][image3]

### Performance considerations

The 'single HOG extraction for entire image' is not implemented due to time constraints.
In its current version the pipeline extracts HOG features from each window image, which is inefficient.
It takes about 2.2 sec to process one frame on my laptop, hardly a real-time speed required in a real car.

I did quick profiling of the pipeline. And after few improvements (batch prediction for all windows is the one worth mentioning)
the most time spent in the pipeline on:




## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

**Alexey Simonov**: 
You're reading it!


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

**Alexey Simonov**: 
I used skimage version of HOG. You can see this in function `features_hog`. You can see an example of its use after its defined in the notebook.

I have found empirically that YUV colorspace for HOG features works better than RGB.
I did not play much with different HOG parameters, such as `orientations`, `pixels_per_cell`, and `cells_per_block`)
as the performance of the classifier was sufficient to subsequently eliminate false positives. 


####2. Explain how you settled on your final choice of HOG parameters.

**Alexey Simonov**: 
I tried various combinations of parameters and...


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

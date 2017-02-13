## Udacity Self-Driving Car Engineer Nanodegree Project 5: Vehicle Detection

---

[//]: # (References)
[image1]: ./examples/hot_windows_example.png
[image2]: ./examples/example_diagnostics1.png
[image3]: ./examples/example_diagnostics2.png
[image4]: ./examples/sliding64.png

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

`VehicleDetector` class defined to keep Vehicle Detection Pipeline parameters, as well as heatmaps in consecutive frames for averaging over video stream. `process_image` is the main working function in this class, which combines calls to functions mentioned above plus the tracking code that averages the heatmaps and produces final detections. 


### Performance considerations

The 'single HOG extraction for entire image' is not implemented due to time constraints.
In its current version the pipeline extracts HOG features from each window image, which is inefficient.
It takes about 2.2 sec to process one frame on my laptop, hardly a real-time speed required in a real car. 

A few simple improvements I have done resulted in better performance, batch prediction for all windows is the one worth mentioning.
A quick profiling exercise of the pipeline shows that the most time spent extracting HOG features for individual windows - about 2 seconds out of total 2.2 secs/frame. So this is the best candidate for improvement, but as it is not the requirement of the project I will leave it for later time, after submission.




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
I used skimage version of HOG. You can see this in function `features_hog`. You can see an example of its use after it is defined in the notebook.

I have found empirically that YUV colorspace for HOG features works better than RGB.
I did not play much with different HOG parameters, such as `orientations`, `pixels_per_cell`, and `cells_per_block`)
as the performance of the classifier was sufficient to subsequently eliminate false positives. 


####2. Explain how you settled on your final choice of HOG parameters.

**Alexey Simonov**: 
I have not tried many combinations of parameters. Playing with colorspace produced good enough classification accuracy to then eliminate false positives in car tracking/heatmaps code, so I stopped experimenting.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

**Alexey Simonov**: 
I trained a `sklearn`'s `LinearSVM`. Please look at the notebook as it's usage is self-documenting.
It achieves 99.1-99.3% accuracy, depending on the color encoding, the best being YUV with YCbCr close second.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**Alexey Simonov**: 
I have generated sliding windows in scales: 128, 96, 64, 48 and 32.
After playing around with various combinations of overlap and reading the slack channel I settled on using two window scales: 128 and 64 px square. They overlap at a fraction of 0.25, giving me 800 window positions to search within the region of interest, as described above. Here is the visualization for 64px windows:

![sliding window tiling at 64 px][image4]

The actual search is done in `search_windows` function which takes an image, window positions and a classifier and returns the list of windows that are classified as 'cars'.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

**Alexey Simonov**: 
`process_diagnostic_views` function in `VehicleDetector` class is a wrapper around `process_image` function. It adds extra information to the final image, detailing steps of the pipeline like window detections, heatmaps etc. Its output is demonstrated next (sorry about the non-matching scales as its not easy to grab the whole image on one screen):

![diagnostic view 1][image2]
![diagnostic view 2][image3]

In the above image the low four images, from left to right from top to bottom are as follows: raw windows identified, raw heatmap, thresholded heatmap, labelled regions.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

**Alexey Simonov**: 

I have taken video with lane lines annotated and processed it using Vehicle Detection pipeline.
Here's a [link to my video result](./project_video_annotated3.mp4). This is the 'diagnostic view' version that also shows intermediate pipeline stages.

Strictly speaking I should have used both pipelines on raw images from the video, but due to lack of time I have run vehicle detection on already annotated video. It did not result in inferior performance of vehicle detection pipeline due potentially to extra visual features superimposed by lane detection.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

**Alexey Simonov**: 

The function `process_image` in `VehicleDetector` is doing heatmap averaging and filtering.

I use `deque` class from python's standard `collections` library to have a buffer of last 10 frame heatmaps.
For each new image I push the new heatmap into it, then average the resulting collection.

My pipeline has two parameters -- the threshold value to pre-filter heatmap at individual frame level. This is simply the minimum number of windows that should overlap at each pixel for it to remain in heatmap. I set this to 1, so only take pixels where we have at least 2 windows. I then scale thresholded individual heatmap to span the range of values from 0 to 255.

Second parameter for heatmap thresholding is applied to the average of individual frame heatmaps. As they are all scaled this is a value from 0 to 255. For example setting it to 64 would filter out all regions in the bottom 25th percentile of the heatmap values.



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Alexey Simonov**: 

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

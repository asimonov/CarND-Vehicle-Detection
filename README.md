## Udacity Self-Driving Car Engineer Nanodegree Project 5: Vehicle Detection

---

[//]: # (References)
[image1]: ./examples/hot_windows_example.png
[image2]: ./examples/diagnostics.png

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

`extract_features` is run on the provided dataset with YCrCb colorspace and then sklearn `StandardScaler()` is calibrated on the full feature set.

I then train `sklearn.LinearSVC` classifier on the full labelled scaled feature set. It gives accuracy of 99.5% on validation set of 20% of the original data.

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
I used `skimage` version of HOG. You can see this in function `features_hog`. You can see an example of its use after it is defined in the notebook.

I have found empirically that YCrCb colorspace for HOG features works better than RGB. And YUV is slightly worse than YCrCb.
I did not play much with different HOG parameters, such as `orientations`, `pixels_per_cell`, and `cells_per_block`)
as the performance of the classifier was sufficient to subsequently eliminate false positives using other techniques.


####2. Explain how you settled on your final choice of HOG parameters.

**Alexey Simonov**: 
I have not tried many combinations of parameters. Playing with colorspace and tweaking averaging/tracking parameters produced good enough classification accuracy to eliminate false positives, so I stopped experimenting.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

**Alexey Simonov**: 
I trained an `sklearn`'s `LinearSVM`. Please look at the notebook as it's usage is self-documenting.
It achieves 99.5% accuracy with YCbCr color encoding. For RGB the accuracy was around 99%.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**Alexey Simonov**: 
I have generated sliding windows in the following scales: 128, 96, 64, 48 and 32.
After playing around with various combinations of overlap and reading the slack channel I settled on using two window scales: 128 and 64 px square. They overlap at a fraction of 0.25, giving me 800 window positions to search within the region of interest, as described above. Here is the visualization for 64px windows:

![sliding window tiling at 64 px][image4]

The actual search is done in `search_windows` function which takes an image, window positions and a classifier and returns the list of windows that are classified as 'cars'.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

**Alexey Simonov**: 
`process_diagnostic_views` function in `VehicleDetector` class is a wrapper around `process_image` function. It adds extra information to the final image, detailing steps of the pipeline like window detections, heatmaps etc. Its output is demonstrated next:

![diagnostic view of the pipeline][image2]

In the above image the low four images, from left to right from top to bottom are as follows: raw windows identified, raw heatmap, thresholded heatmap, labelled regions.

Having the diagnostic view of the pipeline was really beneficial for me to understand where my
classifier performance needs improvement. I started with RGB colorspace and tried both LinearSVC
and SVC with kernels 'linear' and 'rbf'. RBF kernel was too slow so I did not even use it in
window search. Linear SVM was giving 99%+ accuracy, but there were still too many false positives
on individual frames, sometimes sequentially. So I could not eliminate them successfully using
heatmap averaging.
At this point I started to use decision function values (and show them in my diagnostic view as red
circles with radius proportional to decision function value). I was weighting my heatmaps with decision
function values. That improved things. But when false positives appeared in sequential frames I
still could not eliminate them using thresholding only.

That's when I returned to feature engineering. I first tried to use ALL channels for HOG, which
improved things a little, but still not enough.
Then I switched to YUV color space. This was a good step forward as I started to see less false
positives and they started to be more random, not appearing in consequtive frames.
But the main improvement was to switch to YCrCb color space. That practically eliminated all false
positives. Even though it improved classifier validation performance by just 0.1% the result
on the video was perfect. And of course along all these steps I had multiple iterations trying
to fine-tune heatmap averaging and thresholding parameters, but what I have found is the performance
of the classifier is key.


The final project video (just Vehicle Detection) shows full diagnostic view [here](./project_video_annotated_vehicles.mp4) 


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

**Alexey Simonov**: 

First I have taken video with lane lines annotated and processed it using Vehicle Detection pipeline.
It did not work quite well as lane annotations were leading to false positives in car detections.
So I used the raw project video to process vehicle detection only.
Here's the [result](./project_video_annotated_vehicles.mp4). 
This is the 'diagnostic view' version that also shows intermediate pipeline stages.

I have then combined both Lane Detection and Vehicle Detection pipelines to produce [fully annotated version](./project_video_annotated_lanes_and_vehicles.mp4). It does not show diagnostic view for various pipeline stages.


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

The main issues were false positives and performance.
In its current version it was taking me about 50 minutes to process full video. So I had to work with subclips most of the time.
For false positives I investigated the following approaches:

* change sizes/positions of search windows
* change classifier to SVC and play with parameters
* play with parameters of feature extraction, such as color space
* play with heatmap averaging and thresholding parameters

Once I added 'diagnostic views' of the pipeline to the video my workflow became more streamlined as I would see what parameters I need to tweak before the next iteration without trying to figure out from the final output where in the pipeline the problem is best addressed.

The pipeline as it is obviously depends on the input training data. 
It is limited to day/night and weather conditions of the provided dataset.
It is also limited to car makes and colors prevailing at where the training images were taken. So, running it on a video of some other country highway may result in it not detecting cars which are older or dirty, for example.
Also, I would imagine running it on a video of urban environment will produce lots of false positives as some of buildings colors/shapes may look like those of cars to the classifier as it is.


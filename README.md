# CarND-Vehicle-Detection

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle images:

![Car 1]( ./output_images/1.jpeg)   |  ![Car 2]( ./output_images/2.jpeg) |  ![Car 3]( ./output_images/3.jpeg) |  ![Car 4]( ./output_images/4.jpeg)

Non - vehicle images:

![NCar 1]( ./output_images/extra1.png)   | ![NCar 1]( ./output_images/extra1.png) 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Vehicles

Original | Bin patial | Hist patial 
:------:|:------:|:-------:
![Car 1]( ./output_images/1.jpeg)   |  ![Car 2]( ./output_images/1_1_True_binspatial.jpeg) |  ![Car 3]( ./output_images/1_1_True_histpatial.jpeg) 
![Car 2]( ./output_images/2.jpeg)   |  ![Car 2]( ./output_images/1_2_True_binspatial.jpeg) |  ![Car 3]( ./output_images/1_2_True_histpatial.jpeg) 
![Car 3]( ./output_images/3.jpeg)   |  ![Car 2]( ./output_images/1_3_True_binspatial.jpeg) |  ![Car 3]( ./output_images/1_3_True_histpatial.jpeg) 

Non-vehicles

Original | Bin patial | Hist patial 
:------:|:------:|:-------:
![Car 1]( ./output_images/extra1.png)   |  ![Car 2]( ./output_images/1_1_False_binspatial.jpeg) |  ![Car 3]( ./output_images/1_1_False_histpatial.jpeg) 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Vehicles

Original | HOG 
:------:|:------:
![Car 1]( ./output_images/1.jpeg)   |  ![Hog]( ./output_images/1_1_True_hog_All.jpeg) 
![Car 2]( ./output_images/2.jpeg)   |  ![Hog]( ./output_images/1_2_True_hog_All.jpeg) 
![Car 3]( ./output_images/3.jpeg)   |  ![Hog]( ./output_images/1_3_True_hog_All.jpeg) 


Non-vehicles

Original | Hog
:------:|:------:
![Car 1]( ./output_images/extra1.png)   |  ![Hog]( ./output_images/1_1_False_hog_All.jpeg) 

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and use following:

```
  color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  orient = 9  # HOG orientations
  pix_per_cell = 8 # HOG pixels per cell
  cell_per_block = 2 # HOG cells per block
  hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
  spatial_size = (32, 32) # Spatial binning dimensions
  hist_bins = 32    # Number of histogram bins
  spatial_feat = True # Spatial features on or off
  hist_feat = True # Histogram features on or off
  hog_feat = True # HOG features on or off
  y_start_stop = [500, 670] # Min and max in y to search in slide_window()
```

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


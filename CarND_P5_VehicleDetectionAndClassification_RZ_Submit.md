
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Vehicle Detection and Classification

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
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called some_file.py).

I started by reading in all the vehicle and non-vehicle images. Here is an example of one of each of the vehicle and non-vehicle classes:

![png](output_6_1.png)

Color histogram for the sample car is displayed below. 32 bins and the range of (0, 255) is used for computing the histogram.

![png](output_9_0.png)

Spatially-Binned features for the random car using the 'YCrCb' color space is as follows:

![png](output_13_1.png)

I then explored different color spaces `skimage.hog()` parameters. Based on multiple trials, I chose the following parameters for HOG:

```python
# Define HOG parameters
color_space = 'YCrCb' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb
orient = 32
pix_per_cell = 16
cell_per_block = 2

```

Here is the HOG features using the `YCrCb` color space and the above HOG parameters:


![png](output_17_3.png)



2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and numerous trials for feature extraction to get a feel for the parameters. Combination of using all channels of 'YCrCb' color space, spatial binning, color histogram and using all hog channels produced better model accuracy.
For the color space, using three channels helps the model as the model receives more information.
After trying smaller orientations and pixle per cells, I decided to use (32, 32) orientations and (16, 16) pixles per cel to feed more information to the model. Cells per block was kept constant as (2, 2) during different trials.
For spacial binning, I tried 8, 12, 16 and 32 bins. Higher number of bins seemed to result in better accuracy.

![png](output_20_1.png)


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The function 'extract_features' implemented in cell #11 of the notebook uses spatial binning, color histogram and hog features to generate a comprehensive stacked feature set for the images.

The svm is used as the classifier since it provides a good combination of speed and accuracy. The svm is trained in cell #13 of the code. extracted features are mapped to their corresponding lables for the training data set.

The trained svm achieved the accuracy of 0.987 for the testing data.

```python
    15.58 Seconds to train SVC...
    Test Accuracy of SVC =  0.987
    My SVC predicts:  [ 0.  0.  0.  1.  0.  0.  1.  1.  0.  1.]
    For these 10 labels:  [ 0.  0.  0.  0.  0.  0.  1.  1.  0.  1.]
    0.001 Seconds to predict 10 labels with SVC
 ```   



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding window search is implemented in cells #15, 16 and 19. 
I chose three types of windows to search for vehicles: large, medium and small. Parameters for the windows are set as follows:


```python
y_start_stop = [None, None] # Min and max in y to search in slide_window()

# define a function to run through all 3 types of windows
window_x_limits = [[None, None],
                   [40, None],
                   [400, 1280]]

window_y_limits = [[380, 640],
                   [400, 600],
                   [440, 560]]

window_size_src = [(128, 128),
                   (96, 96),
                   (64, 64)]

window_overlap = [(0.6, 0.6),
                  (0.7, 0.7),
                  (0.8, 0.8)]
```

The sliding window function searches for the cars within the specified regions using the corresponding window size and overlaps. 



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

The results on test images is illustrated below:


![png](output_31_2.png)



![png](output_31_3.png)



![png](output_31_4.png)



![png](output_31_5.png)



![png](output_31_6.png)



![png](output_31_7.png)


### Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)



<video width="960" height="540" controls>
  <source src="vehicle_detected.mp4">
</video>


2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  The heat maps are tracked over 10 frames. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

To reduce false positives and combine detected boxes, heat map and thresholding is implemented in cells #21 and 22. The results for the test images is as follows:


![png](output_35_1.png)



![png](output_35_2.png)



![png](output_35_3.png)



![png](output_35_4.png)



![png](output_35_5.png)



![png](output_35_6.png)



---

###Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are still some false positives in the model especially for areas within shades. The models at points faces challenges to fully detect a car which is on the side. Larger set of training data can help the model to perform better. Also, better feature extration using combined color spaces can be helpful.

Also, the model is performs relatively slow which can be a challenge for real-time implementation.

Oncoming traffic is occationally picked up by the model. This can be improved by narrowing the search window to the area in front of the car.






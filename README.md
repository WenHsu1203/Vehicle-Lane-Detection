# Introduction
Develop a software in Python to identify the land boundaires and vehicles in a video from a front-facing camera on a car. THe video is filmed on a highway in California.

# The golas / steps of this project aree the following:

## Lane boundaries detection
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
2. Apply a distortion correction to raw images
3. Use color transforms, gradients, etc., to create a thresholded binary image
4. Apply a perspective transform to rectify binary image ("birds-eye view")
5. Detect lane pixels and fit to find the lane boundary
6. Determine the curvature of the lane and vehicle position with respect to center
7. Warp the detected lane boundaries back onto the original image
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

## Vehicle detection
1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
2. Apply a color transform and append binned color features, as well as histograms of color, to my HOG feature vector.
3. Use the HOG, color and space features as input to train a SVM classifier and use gridSearch to find the optimal parameter.
4. Implement a sliding-window technique and use my trained classifier to search for vehicles in images.
5. Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

# Final Result
**Check out the video below** <br>
[![image](http://img.youtube.com/vi/ejwjEC9vftE/0.jpg)](http://www.youtube.com/watch?v=ejwjEC9vftE "Vehicle Lane Detection")
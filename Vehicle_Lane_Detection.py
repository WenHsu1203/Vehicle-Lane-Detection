import numpy as np
import cv2
import glob
import time
from util import *
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
import pickle
import os

pkl_filename = "svc_model.pkl" 
pkl_filename_2 = "Scaler.pkl" 

class lane_vehcile_detector:
	def __init__ (self):
		# forward camera matrix
		self.M = None
		# inverse camera matrix
		self.invM = None
		# distortion coefficient
		self.dist = None
		# camera matrix
		self.mtx = None

		# twos arrays for warpping
		self.src = np.float32([[585,461],
		                       [200,717],
		                       [1088,704],
		                       [708,459]])
		self.dst = np.float32([[320,0],
		                       [320,720],
		                       [980,720],
		                       [980,0]])
		self.kernel_size = 9
		self.dir_thres = (0.7,1.3)
		self.mag_thres = (30,200)
		self.hls_thres = (59,255)
		self.luv_thres = (150,255)
		self.hsv_thres = (14,100)
		self.lab_thres = (135,255)
		self.svc = None
		self.X_scaler = None

		self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		self.orient = 9  # HOG orientations
		self.pix_per_cell = 8 # HOG pixels per cell
		self.cell_per_block = 2 # HOG cells per block
		self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
		self.spatial_size = (32, 32) # Spatial binning dimensions
		self.hist_bins = 32    # Number of histogram bins
		self.spatial_feat = True # Spatial features on or off
		self.hist_feat = True # Histogram features on or off
		self.hog_feat = True # HOG features on or off
		self.calibrate_camera()
		if not os.path.exists(pkl_filename):
			self.train_vehicle_classifier()
		else:
			self.load_vehicle_classifier()

	def calibrate_camera(self):
		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d points in real world space
		imgpoints = [] # 2d points in image plane.s

		images = glob.glob('camera_cal/calibration*.jpg')

		for fname in images:
		    img = cv2.imread(fname)
		    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		    # Find the chessboard corners
		    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

		    # If found, add object points, image points
		    if ret == True:
		        objpoints.append(objp)
		        imgpoints.append(corners)

		''' 
		  Calibrate the camera
		  Use tge cv2 calibrateCamera function to get the convert(3D to 2D)
		  matrix and distance r
		'''
		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)
		# Camera Matrix for both forward and inverse
		self.M = cv2.getPerspectiveTransform(self.src,self.dst)
		self.invM = cv2.getPerspectiveTransform(self.dst,self.src)

	def train_vehicle_classifier(self):
		cars = glob.glob('vehicles/*.png')
		notcars = glob.glob('non-vehicles/*.png')
		sample_size = 2000
		cars = cars[0:sample_size]
		notcars = notcars[0:sample_size]

		car_features = extract_features(cars, color_space=self.color_space, 
		                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
		                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
		                        cell_per_block=self.cell_per_block, 
		                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
		                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
		notcar_features = extract_features(notcars, color_space=self.color_space, 
		                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
		                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
		                        cell_per_block=self.cell_per_block, 
		                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
		                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)

		# Create an array stack of feature vectors
		X = np.vstack((car_features, notcar_features)).astype(np.float64)

		# Define the labels vector
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

		# Split up data into randomized training and test sets
		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split(
		    X, y, test_size=0.2, random_state=rand_state)
		    
		# Fit a per-column scaler
		self.X_scaler = StandardScaler().fit(X_train)
		with open(pkl_filename_2, 'wb') as file:  
		    pickle.dump(self.X_scaler, file)
		# Apply the scaler to X
		X_train = self.X_scaler.transform(X_train)
		X_test = self.X_scaler.transform(X_test)

		print('Using:',self.orient,'orientations',self.pix_per_cell,'pixels per cell and', self.cell_per_block,'cells per block')
		print('Feature vector length:', len(X_train[0]))
		# Use a linear SVC 
		# self.svc = LinearSVC()
		regressor = SVC()

	    # Create a SVC object

	    # Create a dictionary for the parameter 'C' with a range from 1 to 10 and 
		params = {'C':range(5,6)}

	    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
		ftwo_scorer = make_scorer(fbeta_score, beta=2)

	    # Create the grid search cv object --> GridSearchCV()
	    # Make sure to include the right parameters in the object:
	    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
		grid = GridSearchCV(regressor, params, ftwo_scorer)

		# Check the training time for the SVC
		t=time.time()
		# self.svc.fit(X_train, y_train)
		# Fit the grid search object to the data to compute the optimal model
		grid = grid.fit(X_train, y_train)
	    # Return the optimal model after fitting the data
		self.svc = grid.best_estimator_
		# Save to file in the current working directory
		print("Parameter 'C' is {} for the optimal model.".format(self.svc.get_params()['C']))

		with open(pkl_filename, 'wb') as file:  
		    pickle.dump(self.svc, file)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to train SVC...')
		# Check the score of the SVC
		with open(pkl_filename, 'rb') as file:  
		    model = pickle.load(file)
		print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4))
		# Check the prediction time for a single sample
		t=time.time()


	def load_vehicle_classifier(self):
		with open(pkl_filename, 'rb') as file:  
		    self.svc = pickle.load(file)
		with open(pkl_filename_2, 'rb') as file:  
		    self.X_scaler = pickle.load(file)

	def get_hot_windows(self, img):

		windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 500], xy_window=(110, 96), xy_overlap=(0.8, 0.8))
		windows.extend(slide_window(img, x_start_stop=[None, None], y_start_stop=[500, 600], xy_window=(128, 108), xy_overlap=(0.8, 0.8)))
		windows.extend(slide_window(img, x_start_stop=[None, None], y_start_stop=[600, 650], xy_window=(144, 116), xy_overlap=(0.8, 0.8)))
		windows.extend(slide_window(img, x_start_stop=[None, None], y_start_stop=[650, 700], xy_window=(156, 124), xy_overlap=(0.8, 0.8)))
		windows.extend(slide_window(img, x_start_stop=[None, None], y_start_stop=[700, None], xy_window=(180, 156), xy_overlap=(0.8, 0.8)))
		hot_windows = search_windows(img, windows, self.svc, self.X_scaler, color_space= self.color_space,  
	    	spatial_size= self.spatial_size, hist_bins= self.hist_bins, 
	    	orient= self.orient, pix_per_cell= self.pix_per_cell, 
	    	cell_per_block= self.cell_per_block, 
	    	hog_channel= self.hog_channel, spatial_feat= self.spatial_feat, 
	    	hist_feat= self.hist_feat, hog_feat= self.hog_feat)

		return hot_windows

	def get_heat_map(self, img, hot_windows, threshold):
		heat = np.zeros_like(img[:,:,0]).astype(np.float)
		heat = add_heat(heat, hot_windows)
		heat = apply_threshold(heat, threshold)
		heatmap = np.clip(heat, 0, 255)
		return heatmap


	def process_image(self, img):
		img = img.astype(np.float32)/255
		draw_image = np.copy(img)
		hot_windows = self.get_hot_windows(img)
		heat_thres = 3
		heatmap = self.get_heat_map(img, hot_windows, heat_thres)

	    # Find final boxes from heatmap using label function
		labels = label(heatmap)
		draw_img = draw_labeled_bboxes(np.copy(img), labels)
		img = (draw_img*255).astype(np.uint8)

		'''
		****************************
			  Lane Detection
		***************************
		'''

		undist_image = cal_undistort(img, self.mtx, self.dist)
		gray = cv2.cvtColor(undist_image, cv2.COLOR_RGB2GRAY)
		histogram = np.sum(gray[gray.shape[0]//2:,:], axis=0)
		midpoint = np.int(histogram.shape[0]//2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint
		center = (leftx_base+rightx_base)/2
		offset = abs(center-gray.shape[1]/2)*3.7/abs(rightx_base - leftx_base)
		offset = round(offset, 2)
		warpped_img = warp(undist_image,self.M)
		# def combined_select(img. hls_thres=(0, 255),luv_thres=(0,255),hsv_thres = (0,180), lab_thres = (155,200)):
		combined_binary = combined_select(warpped_img, self.hls_thres, self.luv_thres, self.hsv_thres, self.lab_thres)
		result,rad = fit_polynomial(combined_binary)
		unwarpped_img = unwarp(result,self.invM)
		final = cv2.addWeighted(undist_image, 1, unwarpped_img, 0.7, 0)
		cv2.putText(final, "Radius of Curvature = "+str(rad),(25, 50), cv2.FONT_HERSHEY_SIMPLEX,
		            2.0, (255, 255, 255),2, lineType=cv2.LINE_AA)
		cv2.putText(final, "Vehicle is "+str(offset)+"m left of center",(25, 120), cv2.FONT_HERSHEY_SIMPLEX,
		            2.0, (255, 255, 255),2, lineType=cv2.LINE_AA)
		return final

def main():
	lvd = lane_vehcile_detector()
	from moviepy.editor import VideoFileClip
	white_output = 'test_output_3.mp4'
	clip1 = VideoFileClip("project_video.mp4")#.subclip(30, 40)
	white_clip = clip1.fl_image(lvd.process_image) 
	white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__" :
    main()

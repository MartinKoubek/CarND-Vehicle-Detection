'''
Created on 23. 8. 2017

@author: ppr00076
'''
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from classifier import Classifier
from feature import Feature
from sliding_window import SlidingWindow
from heat_map import HeatMap
from tracking_pipeline2 import TrackingPipeline
from image_show import *
from moviepy.editor import VideoFileClip
import cv2

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

tracking = TrackingPipeline()
SHOW_IMAGES = False
TEST = 0

class VehicleDetection(object):
    def __init__(self):
        self.feature = Feature(None, None, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat,
                        show_debug = False)
        
        self.feature.learn()
        # scaler to X
        scaled_X, self.X_scaler = self.feature.get_scaled_X()
        
        # Define the labels vector
        y = self.feature.get_label_vector()
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        
        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        self.c = Classifier('SVC')
        self.c.run(X_train, y_train)
        print('SVC Accuracy: ',self.c.getAccuracy(X_test, y_test))
        
    def process_image(self, img):
        s = SlidingWindow(x_start_stop=[None, None], 
                          y_start_stop=y_start_stop,
                          classifier = self.c,
                          scaler = self.X_scaler,
                          color_space=color_space, 
                          spatial_size=spatial_size, 
                          hist_bins=hist_bins, 
                          orient=orient, pix_per_cell=pix_per_cell, 
                          cell_per_block=cell_per_block, 
                          hog_channel=hog_channel, spatial_feat=spatial_feat, 
                          hist_feat=hist_feat, hog_feat=hog_feat,
                          show_debug = SHOW_IMAGES)
        
        window_img, box_list = s.run(img)
        
        h = HeatMap(img, 1, show_debug = SHOW_IMAGES)
        h.set_box_list(box_list)
        heat_img = h.get_heat()
        bound_img = h.get_boxed_image(img)
        box_list2 = h.get_boxes()
#         
#         for i in range(10):
        tracking.set_box_list(img, box_list)

        tracking_heat_img = tracking.get_heat()
        tracking_img = tracking.get_boxed_image(bound_img)
        
        if (SHOW_IMAGES):
            plt.close()
            plt.imshow(window_img)
            plt.savefig("output_images/4_windows_image.png")
            
            plt.close()
            plt.imshow(bound_img)
            plt.savefig("output_images/4_boxed_image.png")
            
            plt.close()
            plt.imshow(tracking_heat_img)
            plt.savefig("output_images/4_tracking_heat_image.png")
            
            plt.close()
            plt.imshow(tracking_img)
            plt.savefig("output_images/4_tracking_image.png")


        
#         imageShow(img, \
#                   #img_calibrated, "Calibrated", \
#                   window_img, "windows", \
#                   heat_img, "Heat",
#                   bound_img, "bounded",
#                   tracking_heat_img, "Tracked heat", 
#                   tracking_img, "Tracked")

        return tracking_img
        

if __name__ == '__main__':
    
    detection = VehicleDetection()
    
    
    if (TEST):
        img = mpimg.imread('template_images/bbox-example-image.jpg')
        draw_image = np.copy(img)    
        detection.process_image(draw_image)
    
    else:
        video_output1 = 'project_video.mp4'
        images_output1 = 'video_test_images/frame%03d.png'
        #images_output1 = 'video_test_images/frame%03d.png'
        video_input1 = VideoFileClip(video_output1)#.subclip(10,23)
        processed_video = video_input1.fl_image(detection.process_image)
        processed_video.write_videofile("out_" + video_output1, audio=False)
        #processed_video.write_images_sequence(images_output1)
    
    
    
#     spatial = 32
#     histbin = 32
#     f = ColorFeature(spatial, histbin)
#     f.learn('RGB')
#     car_features, notcar_features, color_scaled_X = f.getFeaturesVector()
#     
#     h = HogFeature(colorspace,
#                    orient,pix_per_cell,
#                    cell_per_block ,hog_channel)
#     h.learn()
#     hog_scaled_X = h.getFeaturesVector()
#     
#     
#     # Define the labels vector
#     y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
#     
#     
#     # Split up data into randomized training and test sets
#     rand_state = np.random.randint(0, 100)
#     color_X_train, color_X_test, color_y_train, color_y_test = train_test_split(
#         color_scaled_X, y, test_size=0.2, random_state=rand_state)
#     
#     hog_X_train, hog_X_test, hog_y_train, hog_y_test = train_test_split(
#         hog_scaled_X, y, test_size=0.2, random_state=rand_state)
#     
#     print('Using spatial binning of:',spatial,
#         'and', histbin,'histogram bins')
#     print('Feature vector length:', len(color_X_train[0]))
#     
#     color_c = Classifier('SVC')
#     color_c.run(color_X_train, color_y_train)
#     print('Color SVC predicts: ',color_c.getAccuracy(color_X_test, color_y_test))
#     
#     print('Using:',orient,'orientations',pix_per_cell,
#     'pixels per cell and', cell_per_block,'cells per block')
#     hog_c = Classifier('SVC')
#     hog_c.run(hog_X_train, hog_y_train)
#     print('HOG SVC predicts: ',hog_c.getAccuracy(hog_X_test, hog_y_test))
#     
#     


    
    print("done")
    
    
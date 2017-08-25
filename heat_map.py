'''
Created on 25. 8. 2017

@author: ppr00076
'''
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

class HeatMap(object):

    def __init__(self, image, threshold, bold = False, show_debug = False):
        self.image = np.copy(image)        
        self.threshold = threshold
        self.bold= bold
        self.heatmap = None
        self.show_debug = show_debug
    
    def set_box_list(self, box_list):
        heat = np.zeros_like(self.image[:,:,0]).astype(np.float)
        for box in box_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        
        self.heatmap_orig = np.clip(heat, 0, 255)
        
#         plt.imshow(self.heatmap_orig)
#         plt.show()
        
        heat[heat <= self.threshold] = 0        
        self.heatmap_treshold = np.clip(heat, 0, 255)
        
        if (self.show_debug):
            plt.close()
            plt.imshow(self.heatmap_orig)
            plt.savefig("output_images/3_heat_image.jpeg")
            plt.close()
            plt.imshow(self.heatmap_treshold)
            plt.savefig("output_images/3_heat_image.jpeg")
            self.show_debug = False
            
#         plt.imshow(self.heatmap_treshold)
#         plt.show()
        
    def get_heat(self):
        return self.heatmap_orig
    
    def get_boxes(self):
        return self.labeled_bboxes()
    
    def get_boxed_image(self, image):
        image = np.copy(image) 
        box_list = self.labeled_bboxes()
        for bbox in box_list: 
            if (self.bold):
                cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)
            else:
                cv2.rectangle(image, bbox[0], bbox[1], (100,100,100), 3)
        return image
        
    def labeled_bboxes(self):
        bbox_list = []
        labels = label(self.heatmap_treshold)
        image = np.copy(self.image)
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            bbox_list.append((bbox))
        # Return the image
        return bbox_list
        
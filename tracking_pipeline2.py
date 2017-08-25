'''
Created on 25. 8. 2017

@author: ppr00076
'''

import math
import numpy as np
import cv2
from heat_map import HeatMap

MAX_IMAGE = 5
TRESHOLD = 7

class TrackingPipeline(object):

    def __init__(self):  
        self.heatmap = None
        self.box_list_history = []
    
    def set_box_list(self, image, box_list):
        if (self.heatmap == None):
            self.heatmap = HeatMap(image, TRESHOLD, bold=True, show_debug = False)
            
        self.box_list_history.append(box_list)
#         for box_hist in self.box_list_history:
        if len(self.box_list_history) > MAX_IMAGE:
            del self.box_list_history[0]
        
        bbox_list = []
        for box_hist in self.box_list_history:
            bbox_list += box_hist
                          
        self.heatmap.set_box_list(bbox_list)

    def get_heat(self):
        return self.heatmap.get_heat()
    
    def get_boxes(self):
        return self.heatmap.get_boxes()
    
    def get_boxed_image(self, image):
        return self.heatmap.get_boxed_image(image)
        
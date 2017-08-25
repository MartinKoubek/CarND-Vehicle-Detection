'''
Created on 24. 8. 2017

@author: ppr00076
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feature import Feature
from image_show import *

class SlidingWindow(object):
    def __init__(self,x_start_stop=[None, None], 
                 y_start_stop=[None, None],
                 xy_window=(96, 96), 
                 xy_overlap=(0.5, 0.5),
                 classifier = None,
                 scaler = None,
                 color_space='RGB', spatial_size=(32, 32),
                 hist_bins=32, orient=9, 
                 pix_per_cell=8, cell_per_block=2, hog_channel=0,
                 spatial_feat=True, hist_feat=True, hog_feat=True,
                 show_debug = False):
                 
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap
        self.classifier = classifier
        self.scaler = scaler
        self.color_space=color_space
        self.spatial_size=spatial_size
        self.hist_bins=hist_bins
        self.orient=orient
        self.pix_per_cell=pix_per_cell 
        self.cell_per_block=cell_per_block
        self.hog_channel=hog_channel
        self.spatial_feat=spatial_feat
        self.hist_feat=hist_feat
        self.hog_feat=hog_feat
        self.show_debug = show_debug
    
    def run(self, img):
        img = np.copy(img)
#         windows = self.slide_window(img, x_start_stop=self.x_start_stop, 
#                                     y_start_stop = self.y_start_stop, 
#                     xy_window=self.xy_window, 
#                     xy_overlap=self.xy_overlap)
#         window_img = self.draw_boxes(img, windows, color=(142, 110, 0), 
#                                      thick=1)
#         
#         on_windows = self.search_windows(img, windows)
#         window_img = self.draw_boxes(window_img, on_windows, color=(85, 142, 0), 
#                                      thick=3)
        #500 , 670
        self.y_start_stop = (380,550)
        scale = 1
        out_img1, box_list1 = self.find_cars(img, scale)
        
        self.y_start_stop = (400,600)
        scale = 1.5
        out_img2, box_list2 = self.find_cars(img, scale)
        
        self.y_start_stop = (400,670)
        scale = 2.0
        out_img3, box_list3 = self.find_cars(img, scale)
        
        if (self.show_debug):
            cv2.imwrite("output_images/2__original_image.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images/2_image1.png", cv2.cvtColor(out_img1, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images/2_image2.png", cv2.cvtColor(out_img2, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images/2_image3.png", cv2.cvtColor(out_img3, cv2.COLOR_RGB2BGR))
            self.show_debug = False
        
        box_list = box_list1 + box_list2 + box_list3
        #bbox_list = [item for sublist in box_list for item in sublist] 
        out_img = self.draw_boxes(img, box_list, thick=3)

#         imageShow(img, \
#                   #img_calibrated, "Calibrated", \
#                   out_img1, "1", \
#                   out_img2, "2" , \
#                   out_img3, "3",
#                   out_img, "All" )
        return out_img, box_list  
    
    # Here is your draw_boxes function from the previous exercise
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
    
    # Define a function that takes an image,
    # start and stop positions in both x and y, 
    # window size (x and y dimensions),  
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    
    def search_windows(self, img, windows):
#                     , color_space='RGB', 
#                     spatial_size=(32, 32), hist_bins=32, 
#                     hist_range=(0, 256), orient=9, 
#                     pix_per_cell=8, cell_per_block=2, 
#                     hog_channel=0, spatial_feat=True, 
#                     hist_feat=True, hog_feat=True):

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)
            #5) Scale extracted features to be fed to classifier
            reshape = np.array(features).reshape(1, -1)
            test_features = self.scaler.transform(reshape)
            #6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
    
    # Define a function to extract features from a single image window
    # This function is very similar to extract_features()
    # just for a single image rather than list of images
    def single_img_features(self, img):    
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)      
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = Feature.bin_spatial(feature_image, size=self.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = Feature.color_hist(feature_image, nbins=self.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(Feature.get_hog_features(feature_image[:,:,channel], 
                                        self.orient, self.pix_per_cell, self.cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = Feature.get_hog_features(feature_image[:,:,self.hog_channel], self.orient, 
                            self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)
    
        #9) Return concatenated array of features
        return np.concatenate(img_features)
    
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, scale):
        
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        
        img_tosearch = draw_img[self.y_start_stop[0]:self.y_start_stop[1],:,:]
        ctrans_tosearch = Feature.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1, n = Feature.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2, n = Feature.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3, n = Feature.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        bbox_list=[]
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
               
                # Get color features
                spatial_features = Feature.bin_spatial(subimg, size=self.spatial_size)
                hist_features = Feature.color_hist(subimg, nbins=self.hist_bins)
     
                # Scale features and make a prediction
                reshape = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.scaler.transform(reshape)    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.classifier.predict(test_features)
                #test_prediction = self.classifier.predict(hog_features)
                
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bbox_list.append(((xbox_left, ytop_draw+self.y_start_stop[0]),(xbox_left+win_draw,ytop_draw+win_draw+self.y_start_stop[0])))
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+self.y_start_stop[0]),(xbox_left+win_draw,ytop_draw+win_draw+self.y_start_stop[0]),(0,0,100),3)
#

                elif (self.show_debug):
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+self.y_start_stop[0]),(xbox_left+win_draw,ytop_draw+win_draw+self.y_start_stop[0]),(100,100,100),1)

                    
        return draw_img, bbox_list


if __name__ == '__main__':
    image = mpimg.imread('template_images/bbox-example-image.jpg')
    s = SlidingWindow()
    window_img = s.run(image)
    plt.imshow(window_img)
    plt.show()
'''
Created on 24. 8. 2017

@author: ppr00076
'''
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
import pickle
import os.path
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

# from color_feature import ColorFeature
# from hog_feature import HogFeature

# color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 9  # HOG orientations
# pix_per_cell = 8 # HOG pixels per cell
# cell_per_block = 2 # HOG cells per block
# hog_channel = 0 # Can be 0, 1, 2, or "ALL"
# spatial_size = (16, 16) # Spatial binning dimensions
# hist_bins = 16    # Number of histogram bins
# spatial_feat = True # Spatial features on or off
# hist_feat = True # Histogram features on or off
# hog_feat = True # HOG features on or off
pickle_file = "features.p"

class Feature(object):
    def __init__(self,cars, notcars, 
                color_space, 
                spatial_size, hist_bins, 
                orient, pix_per_cell, 
                cell_per_block, 
                hog_channel, spatial_feat, 
                hist_feat, hog_feat,
                show_debug = False):
        
        if (notcars == None):
            self.notcars = glob.glob(r'template_images/dataset/non-vehicles_smallset/*/*.jpeg')
        else:
            self.notcars = notcars
        
        if (cars == None) : 
            self.cars = glob.glob(r'template_images/dataset/vehicles_smallset/*/*.jpeg')
        else:
            self.cars = cars

        self.car_features = [] 
        self.notcar_features = []
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

    def learn(self):
        if (os.path.isfile(pickle_file) and not self.show_debug): 
            dist_pickle = pickle.load( open( pickle_file, "rb" ) )
            self.car_features = dist_pickle["car_features"]
            self.notcar_features = dist_pickle["notcar_features"]
        else:
        # Create a list to append feature vectors to
            self.car_features = self.getFeatureAdvance(self.cars, cars = True)
            self.notcar_features = self.getFeatureAdvance(self.notcars, cars = False)
        
            dist_pickle = {}
            dist_pickle["car_features"] = self.car_features
            dist_pickle["notcar_features"] = self.notcar_features
            pickle.dump(dist_pickle, open( pickle_file, "wb" ) )
        

    def get_scaled_X(self):
        X = np.vstack((self.car_features, self.notcar_features)).astype(np.float64)                      
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        self.show_debug = False
        if (self.show_debug):
            plt.close()
            plt.hist(scaled_X)
            plt.savefig("output_images/1_scaled.jpeg")
            self.show_debug = False
                
        return scaled_X, X_scaler
    
    def get_label_vector(self):
        return np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.notcar_features))))
        
    def getFeature(self, imgs):
        # Iterate through the list of images
        features= []
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if self.color_space != 'RGB':
                if self.color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif self.color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif self.color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif self.color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)      
    
            
            if self.spatial_feat == True:
                spatial_features = self.bin_spatial(feature_image, size=self.spatial_size)
                file_features.append(spatial_features)
            if self.hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=self.hist_bins)
                file_features.append(hist_features)
            if self.hog_feat == True:
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:,:,channel], 
                                            self.orient, self.pix_per_cell, 
                                            self.cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = self.get_hog_features(
                        feature_image[:,:,self.hog_channel], 
                        self.orient, 
                        self.pix_per_cell, 
                        self.cell_per_block, 
                        vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features
    
    def getFeatureAdvance(self, imgs, cars):
        # Iterate through the list of images
        features= []
        id = 0
           
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            id += 1
            ctrans_tosearch = self.convert_color(image, conv='RGB2YCrCb')
            
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]
            
            # Compute individual channel HOG features for the entire image
            hog1, hog_image1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=True, vis = self.show_debug)
            hog2, hog_image2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=True, vis = self.show_debug)
            hog3, hog_image3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=True, vis = self.show_debug)
            
            hog_feat1 = hog1.ravel()
            hog_feat2 = hog2.ravel()
            hog_feat3 = hog3.ravel()
            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            # Get color features
            spatial_features = self.bin_spatial(ctrans_tosearch, size=self.spatial_size)
            hist_features = self.color_hist(ctrans_tosearch, nbins=self.hist_bins)
            
            f = np.hstack((spatial_features, hist_features, hog_features))
            
            if ((self.show_debug) and (id < 5)):
                cv2.imwrite("output_images/1_orig_file_" + str(id) + ".jpeg", cv2.cvtColor(ctrans_tosearch, cv2.COLOR_YCrCb2RGB))
                plt.close()
                plt.hist(spatial_features)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) + "_binspatial.jpeg")
                plt.close()
                plt.hist(hist_features)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_histpatial.jpeg")
                plt.close()
                plt.hist(hog_features)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_hogfeature.jpeg")
                plt.close()
                plt.hist(f)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_allfeature.jpeg")
                plt.close()
                plt.imshow(hog_image1)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_hog1.jpeg")
                plt.close()
                plt.imshow(hog_image2)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_hog2.jpeg")
                plt.close()
                plt.imshow(hog_image3)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_hog3.jpeg")
                vis = cv2.addWeighted(hog_image1, 1, hog_image2, 1, 0)
                vis = cv2.addWeighted(vis, 1, hog_image3, 1, 0)
                plt.close()
                plt.imshow(vis)
                plt.savefig("output_images/1_" + str(id) + "_" + str(cars) +"_hog_All.jpeg")
                plt.close()
                

            #file_features.append(hog_features)
            
            features.append(f)
        # Return list of feature vectors
        return features
    
    # Define a function to compute binned color features 
    @staticmethod 
    def bin_spatial(img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features
    
    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    
    # Define a function to return HOG features and visualization
    @staticmethod
    def get_hog_features( img, orient, pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), 
                                      transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), 
                           transform_sqrt=True, 
                           visualise=vis, feature_vector=feature_vec)
            return features, None
      
    @staticmethod  
    def convert_color(img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

if __name__ == '__main__':
    pass
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import tensorflow as tf

# Data preprocessing and evaluation
class DataPreprocess:
    def __init__(self, seismic_location, labels_location):
        self.seismic, self.labels = self.load_data(seismic_location, labels_location)
        self.seismic_resized, self.labels_resized = self.resize_images()

    def load_data(self, seismic, labels):
        seismic = np.load(seismic)
        labels = np.load(labels)
        return (seismic, labels)

    # Resizing the Images
    def resize_images(self):
        resize_dimensions = (64, 176)
        seismic_resized = []
        labels_resized = []

        for i in range(len(self.seismic)):
            #converting input labels to RGB
            img_float32 = np.float32(self.seismic[i])
            new_image = cv.cvtColor(img_float32, cv.COLOR_GRAY2RGB)
            seismic_resized.append(cv.resize(new_image, resize_dimensions))

            #converting output labels to RGB then making it 5 channels
            new_label = cv.resize(self.labels[i], (64, 176))
            img_float32 = np.float32(new_label)
            new_image = cv.cvtColor(img_float32, cv.COLOR_GRAY2RGB)
            #applying one hot encoding
            one_hot_map = []
            palette = [0, 1, 2, 3, 4, 5]
            for colour in palette:
                class_map = tf.reduce_all(tf.equal(new_image, colour), axis=-1)
                one_hot_map.append(class_map)
            one_hot_map = tf.stack(one_hot_map, axis=-1)
            one_hot_map = tf.cast(one_hot_map, tf.float32)
            labels_resized.append(one_hot_map)

        seismic_resized = np.array(seismic_resized)
        labels_resized = np.array(labels_resized)

        return (seismic_resized, labels_resized)

    # generates distribution of pixels based on an array of images
    def generate_hist(images, shape):
        
        #pixels is a 2d array representing the "mean" image. We calculate the mean image by summing up all the values of the images
        #at each index for each image and then dividing by the total number of images (calculated using numImages). 
        #We will use this average image to plot on the histogram and compare distributions.
        pixels = np.zeros(shape)

        num_images = 0

        for img in images:
            pixels += img
            num_images += 1

        #getting the "average" image 
        pixels /= num_images

        # creating a histogram, with greyscale values from 0 to 255
        histogram, bin_edges = np.histogram(pixels, bins = 256, range=(0, 255))

        #returning histogram and bin_edges objects to be used in the plot_histogram method
        return histogram, bin_edges

    def plot_histogram(self, old_image, new_image, name_1, name_2):

        old_hist, old_bin_edges = self.generate_hist(old_image, old_image.shape)
        
        new_hist, new_bin_edges = self.generate_hist(new_image, new_image.shape)

        #calculating the integral to ensure that all pixels are represented
        old_area = np.sum(np.diff(old_bin_edges) *  (old_hist)) # should be around 250000(500 x 500)
        print("Original number of pixels: " + str(old_area))

        new_area = sum(np.diff(new_bin_edges) * new_hist) # should be around 10000(100 x 100)
        print("Resized number of pixels: " + str(new_area)) #comes out to 9960.37
        
        # Creating the plot itself           
        plt.figure()    

        # Titling the plot
        plt.title("Image Pixel Histogram") 

        # X label
        plt.xlabel("Bins Representing Values from [0,255]")

        # Y label
        plt.ylabel("Number of Pixels")
        
        # plotting the histograms themselves
        plt.bar(old_bin_edges[0:-1], old_hist, label=name_1)
        plt.bar(new_bin_edges[0:-1], new_hist, label=name_2)

        plt.legend(loc = 'upper right')

        # displaying the plot
        plt.show()

    # plots two histograms overlayed, will compare the pixel distributions of the old image and the newly resized image 
    def display_histograms(self):
        min = -1
        max = 1
        scaled = 255*(self.seismic-min)/(max-min)

        resized_min = -1
        resized_max = 1
        resized_scaled = 255*(self.seismic_resized-resized_min)/(resized_max-resized_min)

        self.plot_histogram(scaled, resized_scaled, "Original", "Resized")

        resized_hist, bins = self.generate_hist(resized_scaled, resized_scaled.shape)
            
        plt.figure()    

        # Titling the plot
        plt.title("Resized Image Histogram - Upscaled for Visualization") 

        # X label
        plt.xlabel("Bins Representing Values from [0,255]")

        # Y label
        plt.ylabel("Number of Pixels")

        # plotting the histograms themselves
        plt.bar(bins[0:-1], resized_hist, label = "Resized Upscaled")

        plt.legend(loc = 'upper right')

        # displaying the plot - it should almost identical, but the number of pixels should be less
        plt.show()

def main():
    Data = DataPreprocess('data/train/train_seismic.npy', 'data/train/train_labels.npy')
    Data.display_histograms()
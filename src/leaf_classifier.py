# <-------------------------> Import <------------------------->
import cv2 as cv 
import matplotlib.pylab as plt
import numpy as np
from glob import glob
# <---------------------> Internal Import <--------------------->
from DIPlib.intensityTransform import *
from DIPlib.enhancements import *
from DIPlib.fourier import *
from DIPlib.filters.frequency import *
from DIPlib.morphology import *
from DIPlib.features.regions import *

from skimage.exposure import equalize_hist
import skimage.morphology as skmorph

# <-----------------------> Main Script <-----------------------> 
DATABASE_PATH = "input/Leaves/"

if __name__ == "__main__":
    
    input_flies1 = glob(DATABASE_PATH +"1/" + "*")
    input_flies2 = glob(DATABASE_PATH +"2/" + "*")
    input_flies = input_flies1 + input_flies2
    print(input_flies)
    
    for f in input_flies:
            
        # - Read image
        input_img = cv.imread(f)
        rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)

        # plt.imshow(rgb_img)
        # plt.show()
    
        # - Color Radius Segmentation
        gb_diff = rgb_img[:,:,1].astype(float) - rgb_img[:,:,2].astype(float)
        gb_diff = np.clip(gb_diff,0 ,255).astype(np.uint8)
        _, seg_img = cv.threshold(gb_diff ,None ,255 ,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        # - Morsphological Procressing 
        stre = skmorph.disk(11)
        morph_img = cv.erode(seg_img, stre)
        morph_img = removeFragments(seg_img, thresh_ratio=0.05)
        morph_img = fillHoles(morph_img)
        morph_img = cv.morphologyEx(morph_img, cv.MORPH_CLOSE, stre)
        morph_img = fillHoles(morph_img)
        
        _, eccen = regionBasedFeatures(morph_img, "eccentricity")
        # print(eccen[0])
        
        # - Classification Tree 
        if eccen[0] < 0.8:
            leaf_class = "1"
        else:
            leaf_class = "2"
        print(leaf_class)
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_img)
        # plt.subplot(1, 2, 2)
        # plt.title(f"Leaf Class: {leaf_class}")
        # plt.imshow(morph_img, cmap="gray")
        # plt.show()
        
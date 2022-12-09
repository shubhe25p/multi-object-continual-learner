from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from tqdm import tqdm
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()


backSubMOG = cv.createBackgroundSubtractorMOG2()
backSubKNN = cv.createBackgroundSubtractorKNN()
## [capture]

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

fish_centroid_across_frame = []

while True:
    ret, frame = capture.read()

    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMaskMOG = backSubMOG.apply(frame)
    fgMaskKNN = backSubKNN.apply(frame)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    # cv.imshow('Frame', frame)
    #cv.imshow('FG Mask MOG', fgMaskMOG)
    # cv.imshow('FG Mask KNN', fgMaskKNN)

    fish_blobs, numFish = label(fgMaskKNN, return_num=True)
    # print(numFish)
    properties = ['area', 'bbox', 'bbox_area','eccentricity']
    bbox_df = pd.DataFrame(regionprops_table(fish_blobs, properties = properties))
    bbox_df = bbox_df[(bbox_df['eccentricity'] < bbox_df['eccentricity'].max()) & (bbox_df['bbox_area'] > 200)]
    fish_coord = [(row['bbox-0'], row['bbox-1'],row['bbox-2'], row['bbox-3']) for index,row in bbox_df.iterrows()]
    print(np.array(fish_coord).shape)
    fish_centroid_frame=[]
    for blob in (fish_coord):
        fish_centroid_frame.append(np.array([(blob[1]+blob[3]/2.0),(blob[0]+blob[2])/2.0]))
        cv.rectangle(frame, (int(np.float32(blob[1])), int(np.float32(blob[0]))), (int(np.float32(blob[3])), int(np.float32(blob[2]))),(0,255,0))
    cv.imshow('detected blobs', frame)
    # print(np.array(fish_centroid_frame).shape)
    # print(fish_centroid_frame)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
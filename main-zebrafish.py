import copy
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
import cv2
from ObjectTracker import ObjectTracker
from Drawline import draw_line
import numpy as np
import random
def main():
    
    cap = cv2.VideoCapture("./wells/crop-z-00.avi")
    backSubKNN = cv2.createBackgroundSubtractorKNN()

    tracker = ObjectTracker(160, 8, 3,1)
    
    # labels = open('./coco-labels').read().strip().split('\n')
    # net = cv2.dnn.readNetFromDarknet('./yolov3.cfg', './yolov3.weights')
    # layer_names = net.getLayerNames()
    # print(net.getUnconnectedOutLayers(), (layer_names))
    # layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # height, width = None, None
    # writer = None
    
    while(True):
        
        ret, frame = cap.read()
        print(frame.shape)
        fgMaskKNN = backSubKNN.apply(frame)
        fish_blobs, numFish = label(fgMaskKNN, return_num=True)
        properties = ['area', 'bbox', 'bbox_area','eccentricity']
        bbox_df = pd.DataFrame(regionprops_table(fish_blobs, properties = properties))
        bbox_df = bbox_df[(bbox_df['eccentricity'] < bbox_df['eccentricity'].max()) & (bbox_df['bbox_area'] > 200)]
        fish_coords = [(row['bbox-0'], row['bbox-1'],row['bbox-2'], row['bbox-3']) for index,row in bbox_df.iterrows()]
        # if width is None or height is None:
        #     height, width = frame.shape[:2]

        orig_frame = copy.copy(frame)
        
        if not ret:
            break
        
        centers=[]
        for coord in fish_coords:
            center=np.array([int(coord[0]+coord[2]/2),int(coord[1]+coord[3])/2])
            print(coord)
            centers.append(np.expand_dims(center, axis=1))
            # cv2.circle(frame, (int(coord[0]), int(coord[1])), 5, (255,200,0), -1)
            # cv2.circle(frame, (int(coord[2]), int(coord[3])), 5, (10,200,10), -1)
            # cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255,0,0), -1)
            cv2.rectangle(frame, (int(np.float32(coord[1])), int(np.float32(coord[0]))), (int(np.float32(coord[3])), int(np.float32(coord[2]))),(0,255,0))
        cv2.imshow('detected fish with bbox', frame)

        # y_predicted = model.predict(frame)


        # centers = infer_image(net, layer_names, height, width, frame, labels)
        print("Number of coordinates for fish",len(centers))
        
        # if (len(centers) > 0):

        #     tracker.Update(centers)
        #     # print(tracker.objects[0])
        #     prob = random.uniform(0, 1)
        #     draw_line(tracker,frame)
        #     cv2.imshow('Tracking', frame)


        #     print("Prob match ID 1", prob)

            # save_video(writer,frame)

        cv2.imshow('Original', orig_frame)
        
        cv2.waitKey(50)


    # writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
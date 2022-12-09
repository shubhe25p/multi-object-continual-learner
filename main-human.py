import cv2, copy
from Drawline import draw_line, save_video
from helper import infer_image
from ObjectTracker import ObjectTracker

def main():
    
    cap = cv2.VideoCapture("video.mp4")

    tracker = ObjectTracker(160, 8, 3,1)
    
    labels = open('./coco-labels').read().strip().split('\n')
    net = cv2.dnn.readNetFromDarknet('./yolov3.cfg', './yolov3.weights')
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width = None, None
    writer = None
    
    while(True):
        
        ret, frame = cap.read()
        if width is None or height is None:
            height, width = frame.shape[:2]

        orig_frame = copy.copy(frame)
        
        if not ret:
            break

        centers = infer_image(net, layer_names, height, width, frame, labels)
        
        if (len(centers) > 0):

            tracker.Update(centers)

            draw_line(tracker,frame)
            
            cv2.imshow('Tracking', frame)

            save_video(writer,frame)

        cv2.imshow('Original', orig_frame)

        cv2.waitKey(50)


    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
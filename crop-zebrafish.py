import numpy as np
import cv2

# Open the video
cap = cv2.VideoCapture('zebrafish.mp4')

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(w_frame, h_frame)
# Here you can define your croping values
# 1180 796
x,y,h,w = 10,10,100,105

# output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('zebrafish.mp4', fourcc, fps, (w, h))


# Now we start
while(cap.isOpened()):
    ret, frame = cap.read()

    cnt += 1 # Counting frames

    # Avoid problems when video finish
    if ret==True:
        # Croping the frame
        crop_frame = frame[y:y+h, x:x+w]

        # Percentage
        xx = cnt *100/frames
        print(int(xx),'%')

        # Saving from the desired frames
        #if 15 <= cnt <= 90:
        #    out.write(crop_frame)

        # I see the answer now. Here you save all the video
        out.write(crop_frame)

        # Just to see the video in real time          
        cv2.imshow('frame',frame)
        cv2.imshow('croped',crop_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()


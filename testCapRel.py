import numpy as np
import cv2
import time



PROTOCOL = "rtsp"
IP = "192.168.7.57"

USERNAME = "admin"
PASSWORD = "Passw0rd"
PORT = "554"




connection_string = "{}://{}:{}@{}:{}/stream0".format(PROTOCOL, USERNAME, PASSWORD, IP, PORT)   # rtsp://admin:Passw0rd@192.168.7.57:554/stream0
print("Connection String: ", connection_string)

  
fpsframe = 30
  
   
# Create an object to read 
# from camera
video = cv2.VideoCapture(connection_string)


# ip2 = "192.168.7.80"
# video = cv2.VideoCapture(ip2)
   
# We need to check if camera
# is opened previously or not
if (video.isOpened() == False): 
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
output_path = 'C:/Users/AI/Desktop/pose/Video_test4.mp4'

# result = cv2.VideoWriter(output_path,
#                          cv2.VideoWriter_fourcc(*'mp4v'),
#                          fpsframe, size)
    
while(True):
    ret, frame = video.read()
  
    if ret == True: 
        cv2.imshow('Frame', frame)
        #press p to capture
        
        # result.write(frame)

        # to stop the process
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

  
    # Break the loop
    else:
        break
  
# When everything done, release 
# the video capture and video 
# write objects
video.release()
# result.release()
    
# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")
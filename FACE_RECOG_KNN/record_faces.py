#we will be using open Cv for capturing images from web camera

import numpy as np
import cv2

#instanciated a camera object to capture the images ; number represents which  camera you want to use ; 0 is the default value ; 0 for system's web cam
cam = cv2.VideoCapture(0) 


#open cv provides a class for face detection --> haarcascade (to extract the facial features)
##creating a haar-cascade object for face detection
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


##create a placeholder for storing the data
data = [] #all faces will be pushed inside this data list

##current frame number
ix = 0 #flag for how many examples are captured

"""
an infinite loop
calling camera object to read a frame each time by using cam.read()
cam.read() function returns 2 values -> ret and frame
ret is a boolean value ; if camera is returning an object i.e. working properly ret is True else False
frame object should contain the input frame as a numpy matrix
every image is just a collection of 3 RGB components
1 matrix for each color
image is a collection of pixels
for each pixel there are 3 values RGB
combining them we get our image
assuming the ret variable is True we will convert the frame into greyscale cz open cv function for face recognition works for greyscale
"""

while True :
	##retrieve the ret(boolena) and frame from camera
	ret , frame = cam.read()
	if ret==True :
		##if the camera is working fine we will proceed to extract the face
		gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) ##convert BGR image(current frame) to grayscale
		#gray is a grayscale matrix of our frame

		#applying face detection function on the grayscale image
		#this function -> detectMultiScale takes the frame and some other parameters ; detect all the faces and returns an object
		#every object contains the location of image
		#(x,y) and width as w and height as h of the image -> in object

		##apply haar-cascade to detect the faces in the current frame
		##the other parameters 1.3 and 5 are fine tuning parameters
		faces = face_cas.detectMultiScale(gray , 1.3 , 5)

		#go towards every point in the dataset

		##for reach face object we get , we have
		##the corner coords (x,y)
		##and the width and height of the face

		for (x , y , w , h) in faces :
			#all faces are extracted from the original matrix
			#resized
			#and stored in our data list

			##get the face component from the image frame
			face_component = frame[y:y+h , x:x+w , :] # : means takes all the values from RGB
			
			##resize the face image to 50X50X3
			fc = cv2.resize(face_component , (50,50))

			#image is captured after every 10 frame
			#if more than 20 examples are captured we will stop capturing the data

			##store the face data after every 10 frames
			##only if the number of entries are less than 20
			if ix%10 == 0 and len(data) < 50 :
				data.append(fc)

			#a square is drawn on each face
			##for visualization draw a rectangle around the face in the image
			cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 2)
		ix += 1 ##increment the current frame number
		cv2.imshow('frame' , frame ) #just adding an imshow window for visualization meaning whatever image we had is demonstrated in a different window called frame
		

		"""
		waits for some input from the keyboard ; in every 1msec waits for an input ; if id of that input is 27 which stands for escape key
		or if we have collected 50 faces exit the code
		"""

		##if the user presses the escape key(ID : 27)
		##or the number of images hits 20 , we stop
		##recording
		if cv2.waitKey(1) == 27   or len(data) >=50 :##display the frame 
			break 

	else :

		##if the camera is not working , print "error"
		print "error"

##destroy the windows created and save the dataset
cv2.destroyAllWindows()

##convert the data to numpy format
data = np.asarray(data)

##print the shape as a sanity check
print data.shape

##save the data as a numpy matrix in encoded format
np.save('shuja' ,data) #saved in numpy encoded format with name

#We'll run the script for different people
#and store the data into multiple files
import numpy as np 
import cv2

cam  = cv2.VideoCapture(0)

face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

#declare the type of the font to be used on the output window
font  = cv2.FONT_HERSHEY_SIMPLEX

#using numpy load function to import numpy objects
#load the data from the numpy matrices
# convert matrix into linear vector calling reshape function
# gives 50 linear vector
f_01 = np.load('./khyati.npy').reshape(50,50*50*3) #(50,7500) has 50 faces of length 7500 ; each pixel behaves like a feature now
f_02 = np.load('./shuja.npy').reshape(50,50*50*3)
f_03 = np.load('./prafull.npy').reshape(50,50*50*3)
f_04 = np.load('./tanya.npy').reshape(50,50*50*3)

print f_01.shape 

#look-up dictionary to allocate names
names = {
	0 : 'khyati',
	1 : 'shuja' ,
	2 : 'prafull' ,
	3 : 'tanya'
}


#generating labels for data one by one
#total of 40 labels ; first 20 for khyati and others for khyati1
#create matrix to store the labels
labels  = np.zeros((200,1))

#label 0 for class 1 and label 1 for the other class
labels[:50 , :] = 0.0
labels[50:100, : ] = 1.0
labels[100:150, : ] = 2.0
labels[150:, : ] = 3.0

#concatenate the entire data we have
data  = np.concatenate([f_01, f_02 , f_03 , f_04])

print data.shape ,  labels.shape #gives training data and labels

def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())
                       
def knn(x , train , targets  , k = 5 ):  
    m = train.shape[0]
    dist = []
    for ix  in range(m):
        # storing dist from each point in dist list
        dist.append(distance(x ,train[ix]))
        pass
    
    dist = np.asarray(dist)
    indx  = np.argsort(dist)
    #print indx
    #print dist[indx] # distances in sorted order 
    #print labels[indx]
    #retriving top k values 
    sorted_labels  = labels[indx][:k ]
    #print sorted_labels
    #print np.unique(sorted_labels , return_counts=True)# returns tuple with two list  of labels and their count 
    counts  = np.unique(sorted_labels , return_counts=True)
    # first we need to get the index of max count using argmax 
    #then by using that index we will acess the frist list of labels and find label at that index
    #print counts[0][np.argmax(counts[1])]
    return counts[0][np.argmax(counts[1])]
        
while True:

	ret , frame = cam.read()

	if ret == True :
        #convert to grayscale and get the faces
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces  = face_cas.detectMultiScale(gray , 1.3 , 5 )

		for (x , y , w , h) in faces:
			face_component = frame[y:y+h , x:x+w , :]
			fc = cv2.resize(face_component, (50,50))

            #instead of storing the data this much we will try to recognize
            #after processing the image and rescaling
            #flatten() creates a matrix to a linear vector
            #vectr , training data , labels
            #this function returns a label value
            #check in dictionary lab label corresponds to which label
			lab = knn(fc.flatten() , data , labels )


            #label object is converted to integer -> key to dictionary name ; stored in text value
			text = names[int(lab)]

            #text generated appears on frame
			cv2.putText(frame , text , (x,y) , font , 1,(255,255,0) , 2)

			cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,0,255), 2)

		cv2.imshow('face recognition' , frame)

		if cv2.waitKey(1)  ==  27:
			break  

	else :
		print 'error\n'


cv2.destroyAllWindows()
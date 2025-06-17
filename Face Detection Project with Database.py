import cv2
import os

dataset="Dataset"
name="Arjun_face"

path=os.path.join(dataset,name) # WE ARE CREATING A PATH BY JOINING THE DATASET ALONG WITH NAME WE DESIRE TO NAME IT WHEREIN IT CREATES Dataset/Arjun_face

if not os.path.isdir(path): # WE ARE CHECKING IF THERE IN ANY PATH THAT IS CREATED FOR THE SAME PATH NAME THAT IS Dataset/Arjun_face
    os.mkdir(path) # IF NO PATH IS CREATED THIS WILL LINE CREATE A PATH USING MRDIR COMMAND FOR THE SAME PATH NAME WHICH IS Dataset/Arjun_face

(width,height)= (130,100) # DEFINING THE IMAGE SIZE FOR THE IMAGE TO BE STORED

cam=cv2.VideoCapture(0)

#INITIALIZING THE COUNT TO TRACK THE NUMBER OF IMAGES TAKEN
alg="haarcascade_frontalface_default.xml"
haar_algorithm=cv2.CascadeClassifier(alg)
count=1
while count<181:
    print(count)
    _,img=cam.read()
    grayscaleimage=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coordinates=haar_algorithm.detectMultiScale(grayscaleimage,1.3,4)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0), 2)
        FaceOnlyImage=grayscaleimage[y:y+h,x:x+h]
        FaceOnlyResizeImage=cv2.resize(FaceOnlyImage,(width,height))
        cv2.imwrite("%s/%s.jpg" % (path,count),FaceOnlyResizeImage)
        count+=1
    cv2.imshow("Live Face Detection", img) # IT IS OUTSIDE THE LOOP BECAUSE THE THE FOR LOOP IS ONLY FOR MARKING THE RECTANGLE,TAKING,RESIZING AND SAVING THE PICTURE ONLY OF OUR FACES BUT DURING
    key=cv2.waitKey(10) # BUT DURING OUR LIVE FEED WE WANT THE PICTURE OF OUR WHOLE FACE 
    if key==27:
        break
print("Image captured Successfully!")
cam.release()
cv2.destroyAllWindows()

        


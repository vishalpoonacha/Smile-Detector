import cv2

smile_detector_file = 'smile_detector.xml'
face_detector_file = 'face_detector.xml'


smile_detector = cv2.CascadeClassifier(smile_detector_file)
face_detector = cv2.CascadeClassifier(face_detector_file)

webcam = cv2.VideoCapture(0)




while True:

    successful_frame_read,frame = webcam.read()

    if not successful_frame_read:
        break

    grayscale_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face = face_detector.detectMultiScale(grayscale_image)
    

    for (x,y,w,h) in face:
         #draw all the rectangles around the smile
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Get the sub frame (using numpy N-dimentional array slicing)
        the_face= frame[ y:y+h ,x:x+w ]
       
        grayscale_face_image = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        smile = smile_detector.detectMultiScale(grayscale_face_image,scaleFactor=1.7,minNeighbors=20)
         
         #find all the smiles in the face
        #for (x_,y_,w_,h_) in smile:

             

            #draw all the rectangles around the smile
            # cv2.rectangle(the_face,(x_,y_),(x_ + w_,y_ + h_),(50,50,200),2)
        #label this face as smiling
        if len(smile) > 0:
            cv2.putText(frame,'smiling',(x,y+h+40),fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))









    
    cv2.imshow("smile detector",frame)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
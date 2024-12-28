import cv2
from align import face_alignment

def face_detection_with_image(path_align,path_save):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    img = cv2.imread(path_align)

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

    crop_img = img[y:y+h, x:x+w]
    resized = cv2.resize(crop_img, (96,96))

    #cv2.imshow('Gray', resized)
    cv2.imwrite(path_save, resized)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def face_align_with_face(original_path,align_path):
    face_alignment(original_path)
    face_detection_with_image(align_path)


import facenet_keras
import cv2
import numpy as np
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from align import rotation_detection_dlib
import pickle
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model1 = facenet_keras.facenet()
model1.load_weights('weights.h5')

names = ['Angelina_Jolie','Anuj_Ghugarkar','Bill_Gates','David_Beckham','Jackie_Chan','Omkar_Ghugarkar','Serena_Williams','Tiger_Woods','Tom_Cruise','Unknown']

for i in range(1, 12):
    imageName = "../Data/Test_Images/test" + str(i) + ".jpg"
    print(imageName)
    image = cv2.imread(imageName)
    # Resizing image to fit window
    scale_percent = 80  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img = rotation_detection_dlib(resized_image, 0, show=False)
    # using mtcnn for face detection
    detector = MTCNN()
    # using detect faces function to retrive box, confidence and landmarks of faces
    results = detector.detect_faces(img)
    # if face not detected just skip the image
    if (results == []):
        print("Face not detected")
        continue
    print("1. Face Detected from image")
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face

    face = img[y1:y2, x1:x2]
    gray_img = cv2.imread(imageName,0)
    ret, thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(thresh1, scaleFactor=1.05, minNeighbors=5)
    print(faces)
    if len(faces) == 0:
        print("No Face")
        continue
    else:
        print("2. Face extracted")
        #cv2.imshow('image',thresh1)
    # resized for FaceNet model
    face_pixels = cv2.resize(face, (96, 96))
    face_pixels = face_pixels.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = face_pixels/255

    samples = expand_dims(face_pixels, axis=0)
    # Face embeddings collected
    yhat = model1.predict(samples)
    print("3. Face Embeddings Collected")
    # Loading FaceEmbedding model file
    filename = 'finalized_model.sav'
    prediction_model = pickle.load(open(filename, 'rb'))

    # comparing the embeddings
    #yhat_class = prediction_model.predict(yhat)
    # Retrieving the probability of the prediction
    yhat_prob = prediction_model.predict_proba(yhat)
    yhat_prob = np.reshape(yhat_prob,(10,1))
    print(yhat_prob.shape)
    print("4. Predicting class and probability done")
    yhat_class = np.argmax(yhat_prob,axis=0)
    class_index = yhat_class
    print("Index",class_index)
    print(yhat_prob)
    class_probability = yhat_prob[yhat_class,0] * 100
    print(class_probability)
    print(class_probability.shape)
    print('Prediction Probablity:')
    print(int(class_probability))
    # setting threshold based on probability
    if (class_probability > 25):
        # print("Name:",names[class_index])
        cv2.putText(img, names[int(class_index)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        print( names[int(class_index)])
        cv2.imshow("Output", img)
        cv2.waitKey(0)
    else:
        # print("Person not matched")
        cv2.putText(resized_image, "unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Output", resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()


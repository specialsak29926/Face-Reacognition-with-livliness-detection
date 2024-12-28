import facenet_keras
import cv2
import numpy as np
from numpy import expand_dims

#from align import rotation_detection_dlib
import pickle

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

model1 = facenet_keras.facenet()

model1.load_weights('weights.h5')

names = ['Angelina_Jolie','Anuj_Ghugarkar','Bill_Gates','David_Beckham','Jackie_Chan','Omkar_Ghugarkar','Serena_Williams','Tiger_Woods','Tom_Cruise','Unknown']

while(True):
    ret, frame = video_capture.read()
    image = frame
    # Resizing image to fit window
    scale_percent = 80  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #img = rotation_detection_dlib(resized_image, 0, show=False)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(thresh1, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if len(faces) == 0:
        print("No Face")
        continue
    face = img[y:(y + h), x: (x + w)]
    # resized for FaceNet model
    face_pixels = cv2.resize(face, (96, 96))
    face_pixels = face_pixels.astype('float32')

    face_pixels = face_pixels/255

    samples = expand_dims(face_pixels, axis=0)
    # Face embeddings collected
    yhat = model1.predict(samples)
    # Loading FaceEmbedding model file
    filename = 'finalized_model.sav'
    prediction_model = pickle.load(open(filename, 'rb'))

    # comparing the embeddings
    #yhat_class = prediction_model.predict(yhat)
    # Retrieving the probability of the prediction
    yhat_prob = prediction_model.predict_proba(yhat)
    yhat_prob = np.reshape(yhat_prob,(10,1))
    print(yhat_prob)

    yhat_class = np.argmax(yhat_prob,axis=0)
    class_index = yhat_class

    class_probability = yhat_prob[yhat_class,0] * 100

    # setting threshold based on probability
    if (class_probability > 25):

        cv2.putText(img, names[int(class_index)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)


    else:
        cv2.putText(img, "unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()


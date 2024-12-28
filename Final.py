from livenessmodel import get_liveness_model
import facenet_keras
import cv2
import numpy as np
from numpy import expand_dims
import pickle

model1 = facenet_keras.facenet()
model1.load_weights('weights.h5')

names = ['Angelina_Jolie','Anuj_Ghugarkar','Bill_Gates','David_Beckham','Jackie_Chan','Omkar_Ghugarkar','Serena_Williams','Tiger_Woods','Tom_Cruise','Unknown']

model = get_liveness_model()

font = cv2.FONT_HERSHEY_DUPLEX

# load weights into new model
model.load_weights("../Data/model/model.h5")
print("Loaded model from disk")

process_this_frame = True
input_vid = []

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

result = True
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(result):

    if len(input_vid) < 24:

        ret, frame = video_capture.read()

        liveimg = cv2.resize(frame, (100,100))
        liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
        input_vid.append(liveimg)

    else:
        ret, frame = video_capture.read()

        liveimg = cv2.resize(frame, (100,100))
        liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
        input_vid.append(liveimg)
        inp = np.array([input_vid[-24:]])
        inp = inp/255
        inp = inp.reshape(1,24,100,100,1)
        pred = model.predict(inp)
        input_vid = input_vid[-25:]

        if pred[0][0] > .99:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            if len(faces) == 0:
                print("No Face")
                continue

            for x, y, w, h in faces:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            face = frame[y:y+h, x:x+w]

            face_pixels = cv2.resize(face, (96, 96))
            face_pixels = face_pixels.astype('float32')

            samples = expand_dims(face_pixels, axis=0)
            # Face embeddings collected
            yhat = model1.predict(samples/255)
            print("3. Face Embeddings Collected")
            # Loading FaceEmbedding model file
            filename = 'finalized_model.sav'
            prediction_model = pickle.load(open(filename, 'rb'))

            yhat_prob = prediction_model.predict_proba(yhat)
            yhat_prob = np.reshape(yhat_prob, (9, 1))
            print(yhat_prob.shape)
            print("4. Predicting class and probability done")
            yhat_class = np.argmax(yhat_prob, axis=0)
            class_index = yhat_class
            print("Index", class_index)
            print(yhat_prob)
            class_probability = yhat_prob[yhat_class, 0] * 100
            print(class_probability)
            print(class_probability.shape)
            print('Prediction Probablity:')
            print(int(class_probability))

            if (class_probability > 25):
            # print("Name:",names[class_index])
                cv2.putText(frame, names[int(class_index)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
                print(names[int(class_index)])
            else:
            # print("Person not matched")
                cv2.putText(frame, "unknown", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'WARNING!', (frame.shape[1] // 2, frame.shape[0] // 2), font, 1.0, (0, 0, 255), 1)

    # Display the liveness score in top left corner
        cv2.putText(frame, str(pred[0][0]), (20, 20), font, 1.0, (255, 255, 0), 1)
    # Display the resulting image
        cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.VideoCapture(0).release()
cv2.destroyAllWindows()

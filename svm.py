import facenet_keras
import cv2
import numpy as np
from numpy import savez_compressed
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from align import rotation_detection_dlib
import pickle
from sklearn.metrics import accuracy_score

model1 = facenet_keras.facenet()

X = np.zeros((100,96,96,3))
Y_svm = np.zeros((100,1))
model1.load_weights('weights.h5')
image_dir = "../Data/Cropped_faces"
model1.summary()
count = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        list = file.split('.')
        print(list[0])
        count = int(list[0])
        if count <=80:
            path = os.path.join(root, file)
            img = cv2.imread(path, 1)
            cv2.waitKey(0)
            X[count ,:,:,:] = img
            y_value = int(count/9)
            Y_svm[count,0] = y_value
            print(count," ", y_value)
        else:
            path = os.path.join(root, file)
            img = cv2.imread(path, 1)
            cv2.waitKey(0)
            X[count, :, :, :] = img
            y_value = 9
            Y_svm[count, 0] = y_value
            print(count, " ", y_value)
print(X[0:5,:,:,:])

X_svm = model1.predict(X/255)
print(X_svm[0:5,:])
X_svm,Y_svm = shuffle(X_svm,Y_svm)
X_train, X_test, y_train, y_test = train_test_split(X_svm, Y_svm, test_size=0.3)

print(X_svm.shape)
print(Y_svm.shape)
savez_compressed('face_detection.npz', X_train, y_train, X_test, y_test)

# fit model
model2 = SVC(kernel='rbf',probability=True, max_iter=1000000,C=2,gamma=1)
model2.fit(X_svm, Y_svm)

#Saving Model
filename = 'finalized_model.sav'
pickle.dump(model2, open(filename, 'wb'))
# predict
yhat_train = model2.predict(X_train)
yhat_test = model2.predict(X_test)
# score
score_train = accuracy_score(y_train, yhat_train)
score_test = accuracy_score(y_test, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


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
    print("2. Face extracted")
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
    yhat_prob = np.reshape(yhat_prob,(9,1))
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
        cv2.imshow("Output", img)
        cv2.waitKey(0)
    else:
        # print("Person not matched")
        cv2.putText(resized_image, "unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Output", resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

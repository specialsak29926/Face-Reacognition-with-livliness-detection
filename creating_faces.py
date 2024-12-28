import facenet_keras
import cv2
import numpy as np
import os
from align import face_alignment
from Face_detection_with_provided_img import face_detection_with_image
import time
#model = facenet_keras.facenet()

#model.load_weights('weights.h5')

#model.summary()
BASE_DIR = "Final_images"
Names = ['Angelina_Jolie','Anuj_Ghugarkar','Bill_Gates','David_Beckham','Jackie_Chan','Omkar_Ghugarkar','Serena_Williams','Tiger_Woods','Tom_Cruise']
save_path1 = 'Final_Align_images'
save_path2 = 'Final_faces'
count = 81
name = 'Unknown'
image_dir = os.path.join(BASE_DIR, name)
for root, dirs, files in os.walk(image_dir):
    for file in files:
      path_original = os.path.join(root, file)
      align_save = os.path.join(save_path1,str(count))
      face_save = os.path.join(save_path2,str(count))
      face_alignment(path_original, align_save + ".png")
      time.sleep(1)
      face_detection_with_image(align_save + ".png", face_save + ".png")
      count = count + 1

#img = cv2.imread(path, 1)
#img = cv2.resize(img, (96,96))
#x_train = np.array([img])
#y = model.predict(x_train)

#print(y.shape)

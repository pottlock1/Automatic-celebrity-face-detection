import mtcnn
import numpy as np
from imutils import paths
from datetime import datetime
import cv2
import os
from src.insightface.src.common import face_preprocess




class CropFaces:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_images_path = r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_images'

    def cropfaces(self):
        path_images = list(paths.list_images(self.data_path))
        current = 0
##################################### detecting and cropping faces wid mtcnn ###########
        detector = mtcnn.MTCNN()
        for path_image in path_images:
            name = path_image.split(os.path.sep)[-2]
            img = cv2.imread(path_image)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                faces = detector.detect_faces(img)
            except:
                continue

            for face in faces:
                print('Cropping face from image {}/{}'.format(current, len(path_images)))
                box = face['box']
                keypoints = face['keypoints']
                landmarks = np.array([keypoints['left_eye'][0], keypoints['right_eye'][0], keypoints['nose'][0],
                                      keypoints['mouth_left'][0], keypoints['mouth_right'][0],
                                      keypoints['left_eye'][1], keypoints['right_eye'][1], keypoints['nose'][1],
                                      keypoints['mouth_left'][1], keypoints['mouth_right'][1]
                                      ]).reshape((2, 5)).T
                nimg = face_preprocess.preprocess(img, box, landmarks, image_size= '112, 112')

############################## saving the cropped image in corresponding folder ######################
                dtstring = str(datetime.now().microsecond)
                if not (os.path.exists(os.path.join(self.train_images_path, name))):
                    os.mkdir(os.path.join(self.train_images_path, name))

                try:
                    cv2.imwrite(os.path.join(self.train_images_path, name, '{}.jpg'.format(dtstring)), nimg)
                except:
                    continue
                current += 1

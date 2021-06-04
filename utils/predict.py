import pickle
from imutils import paths
from tensorflow.keras.models import load_model
import cv2
import mtcnn
from src.insightface.src.common import face_preprocess
import numpy as np
from src.insightface.deploy import face_model
from datetime import datetime
import os

class Predict:
    def __init__(self):
        self.pred_img_path = r'/predicted_images'
        self.model = load_model(r'/train_results/model.h5')
        self.le = pickle.loads(open(r'/train_results/le.pickle', 'rb').read())
        self.embedding_model = r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\src\insightface\models\model-y1-test2\model,0'

    def predict(self, test_image_dir):
        test_image_paths = list(paths.list_images(test_image_dir))
        detector = mtcnn.MTCNN()
        emb_model = face_model.FaceModel(image_size='112, 112', model=self.embedding_model, threshold=1.24, det=0)

        current = 0
        for test_image_path in test_image_paths:
            print('predicting for image {}..'.format(current))
            img = cv2.imread(test_image_path)
            faces = detector.detect_faces(img)
            for face in faces:
                box = face['box']
                keypoints = face['keypoints']
                landmarks = np.array([keypoints['left_eye'][0], keypoints['right_eye'][0], keypoints['nose'][0],
                                      keypoints['mouth_left'][0], keypoints['mouth_right'][0],
                                      keypoints['left_eye'][1], keypoints['right_eye'][1], keypoints['nose'][1],
                                      keypoints['mouth_left'][1], keypoints['mouth_right'][1]
                                      ]).reshape((2, 5)).T
                nimg = face_preprocess.preprocess(img, bbox= box, landmark = landmarks,image_size = '112, 112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))


                face_embedding = np.array(emb_model.get_feature(nimg)).reshape(1, -1)

                pred = self.model.predict(face_embedding)
                pred = pred.flatten()
                label = np.argmax(pred)
                if label == 0:
                    celeb = 'amitabh'
                elif label == 1:
                    celeb = 'deepika'
                elif label == 2:
                    celeb = 'emma'
                else:
                    celeb = 'shroud'
                prob = pred[label]

                img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=(0, 255, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.putText(img, (celeb +'|'+str(round(prob*100, 2)) +'%'), org = (box[0], box[1]),fontFace = font,fontScale=1, color=(0, 255, 0), thickness=5)
                dtstring = str(datetime.now().microsecond)

                try:
                    cv2.imwrite(os.path.join(self.pred_img_path, '{}.jpg'.format(dtstring)), img)
                    print('Result saved for image {}!!'.format(current))
                    current += 1
                except:
                    continue
        print('Predictions process is complete successfully!!')
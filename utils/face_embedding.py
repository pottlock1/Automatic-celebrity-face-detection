import numpy as np
from imutils import paths
import cv2
import os
from src.insightface.deploy import face_model
import pickle





class GenerateEmbedding:
    def __init__(self, train_image_path):
        self.train_image_path = train_image_path
        self.image_size = '112, 112'
        self.model = r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\src\insightface\models\model-y1-test2\model,0'
        self.threshold = 1.24
        self.det = 0

    def genfaceembedding(self):
        image_paths = list(paths.list_images(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_images'))
        embedding_model = face_model.FaceModel(image_size = self.image_size, model = self.model,threshold= self.threshold, det=self.det)

        embeddings = []
        names = []
        total = 0
        for i, image_path in enumerate(image_paths):
            print('processing image {}/{}'.format(i, len(image_paths)))
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            face_embedding = embedding_model.get_feature(img)
            name = image_path.split(os.path.sep)[-2]

            embeddings.append(face_embedding)
            names.append(name)
            total += 1

        print(total, ' images has been embedded!!')
        train_data = {'embeddings': embeddings, 'names': names}

################ Dumping embeddings in pickle format ###############################
        f = open(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_dumps\embeddings.pickle', 'ab')
        pickle.dump(train_data, f)
        f.close()
        print('image embeddings hve been saved!!')

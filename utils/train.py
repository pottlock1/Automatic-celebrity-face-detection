import numpy as np
from utils.model_builder import Model
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold



class TrainingOnEmbeddings:
    def __init__(self):
        self.train_data = pickle.loads(open(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_dumps\embeddings.pickle', 'rb').read())
        self.history = {'acc' : [], 'val_acc' : [], 'loss' : [], 'val_loss' : []}
    def modeltraining(self):
        le = LabelEncoder()
        oh = OneHotEncoder()
        labels = le.fit_transform(self.train_data['names']).reshape(-1, 1)
        no_classes = len(np.unique(labels))
        embeddings = np.array(self.train_data['embeddings'])
        oh_labels = oh.fit_transform(labels).toarray()

########## Defining the model ##########################################
        epochs = 50
        batch_size = 2
        input_shape = embeddings.shape[1]

        softmax = Model(input_shape = (input_shape,), no_labels= no_classes)
        model = softmax.build()
################ creating Kfold ##################
        cv = KFold(n_splits=3, random_state=69, shuffle=True)

        for train_idx, val_idx in cv.split(embeddings):
            x_train, x_val, y_train, y_val = embeddings[train_idx], embeddings[val_idx], oh_labels[train_idx], oh_labels[val_idx]
            # y_train, y_val = y_train.reshape(-1, 1), y_val.reshape(-1, 1)
            his = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_val, y_val))
            self.history['acc'] += his.history['accuracy']
            self.history['val_acc'] += his.history['val_accuracy']
            self.history['loss'] += his.history['loss']
            self.history['val_loss'] += his.history['val_loss']

        print('model has been trained successfully!!')
        print('The accuracy of the model on training data and validation data = {} and {}'.format(self.history['acc'][-1], self.history['val_acc'][-1]))
        print('Final loss of the model on train and validation data = {} and {}'.format(self.history['loss'][-1], self.history['val_loss'][-1]))

############## Plotting accuracy and loasses #############################
        # result = pd.DataFrame(self.history)
        # result.plot()

####################### Saving model and encoder for further prediction ######################
        model.save(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_results\model.h5')
        f = open(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_results\le.pickle', 'wb')
        pickle.dump(le, f)
        f.close()
        print('Model and encoder has been saved!!')

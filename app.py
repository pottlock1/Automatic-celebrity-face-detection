from utils.image_downloader import CollectCelebImages
from utils.face_crop import CropFaces
from utils.face_embedding import GenerateEmbedding
from utils.train import TrainingOnEmbeddings
from utils.predict import Predict
from tkinter import *
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import cv2


def download_images():
    celeb_name1 = celeb1.get()
    # celeb_name2 = celeb2.get()
    # celeb_name3 = celeb3.get()
    downloader = CollectCelebImages(celeb_name1)
    downloader.download()

def crop_faces():
    crop = CropFaces('celeb_images')
    crop.cropfaces()

def make_embeddings():
    embeddings = GenerateEmbedding('train_images')
    embeddings.genfaceembedding()

def train():
    training = TrainingOnEmbeddings()
    training.modeltraining()

def predict():
    prediction = Predict()
    prediction.predict(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\test_images')

def predict_single():
    prediction = Predict()
    image_path = filedialog.askopenfilename(initialdir = os.getcwd(), title = 'Select Image', filetypes = (('JPG File','*.jpg'), ('PNG File', '*.png'), ('All Files', '*.*')))
    img = prediction.predict_single(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((500, 400))
    img = ImageTk.PhotoImage(img)
    img_label.configure(image = img)
    img_label.image = img

root = Tk()
root.geometry('1000x800')
root.title("Celebrity's Face Detection")
label = Label(root, text = 'Enter the name of a celebrity', bd = 5, bg = 'black', fg = 'white', font = ('times new roman', 20, 'bold'))
label.place(relx = 0.5, rely = 0.05, anchor = CENTER)

celeb1 = Entry(root, width = 15, bd = 5, font = ('arial', 10))
celeb1.place(x = 8, y = 100)


# celeb2 = Entry(root, width = 15, bd = 5, font = ('arial', 10))
# celeb2.place(x = 185, y = 100)
#
# celeb3 = Entry(root, width = 15, bd = 5, font = ('arial', 10))
# celeb3.place(x = 360, y = 100)

b1 = Button(root, text = 'Download', bd = 2, bg = 'red', font = ('times new roman', 10, 'bold'), width = 15,command = download_images)
b1.place(x = 186, y = 150)

b2 = Button(root, text = 'Crop faces', bg = 'blue', font = ('times new roman', 10, 'bold'), width = 15, command = crop_faces)
b2.place(x = 8, y = 250)

b3 = Button(root, text = 'Make embeddings', bg = 'blue', font = ('times new roman', 10, 'bold'), width = 15, command = make_embeddings)
b3.place(x = 185, y = 250)

b4 = Button(root, text = 'Train model', bg = 'blue', font = ('times new roman', 10, 'bold'), width = 15, command = train)
b4.place(x = 360, y = 250)

img_label = Label(root)
img_label.place(x = 150, y = 300)

b5 = Button(root, text = 'Browse image',bg = 'green', font = ('times new roman', 10, 'bold'), width = 15, command = predict_single)
b5.place(x = 10, y = 550)

root.mainloop()








# cropped_images = CropFaces(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\celeb_images')
# cropped_images.cropfaces()

################# Embedding call
# embeddings = GenerateEmbedding(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\train_images')
# embeddings.genfaceembedding()

################## Training call
# training = TrainingOnEmbeddings()
# training.modeltraining()

# ################# Plotting of the results
# result = pd.DataFrame(training.history)
# result.plot()

############### Making predictions
# prediction = Predict()
# prediction.predict(r'C:\Users\PIYUSH KUMAR\PycharmProjects\celeb_face_detection\test_images')
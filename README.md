# Automatic-celebrity-face-detection
Downloads and trains a model to detect the faces of a given celebrity...

### How to run...
* Make and activate an anaconda environment
```sh
conda create -n env_name
conda activate env_name
```
* Install requirements.txt with following command
```sh
pip install -r requirements.txt
```
* Run app.py in console
```sh
python app.py
```
* Enter the name of your fevorite celebrity and click on 'download' button to download it's images from web
* After downloading, click on 'Crop faces' to crop the faces from downloaded images
* Then click on 'Make embeddings' button to make the embeddings of copped images
* Then click on 'Train' to train a keras model
* The trained model will be saved in the 'train_results' folder in .h5 format
* Click on 'Browse image' button to select an image from local and the result will be shown in the same window

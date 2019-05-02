# Toronto Streetscore: Predicting Perceived Street Safety Using CNN

### Overview
This project aims to build an explainable machine learning model that could predict people's perceived safety toward streets views.
We apply deep Convolutional Neural Network (CNN) techniques on street view images fetched using Google Street View API.


### codes/    

#### Data Exploration:
* PlacePulse1.0_explore.ipynb

Explores the data structure of the original data - PlacePulse1.0 used by the Streetscore project. 

* Streetscore_create_target_label.ipynb

Generates a binary 'safety' label.
 
#### Data Preparation:
* Boston_Stratified_Sampling.ipynb

Stratified sampling 20,000 samples from the Boston data predicted by the Streetscore project considering two factors at the same time: 1). Portion of zone classes of Toronto and 2). Portion of Target “safety” variable.

* boston_split_train_test.ipynb

Splits the downsampled 20,000 boston samples into 80 : 20 as training and test sets, so as to use the trainings and test sets to fetch images.

* fetchimage.py

Fetches images using Google Street View Static API

*  cropimage.py

Crops out the Google Logo on each image

*  check_fetched_images_and_merge_with_target.ipynb

Checks how many images were fetched and merges the fetched images with target variable. You should use this script if you are saving all Boston images into one folder. 

*  merge_fetched_train_test_image_with_target.ipynb

Merges the fetched train / test images with target variable. You should use this script if you are saving Boston images into two  (train and test) seperate folders. 

*  Toronto_sample.ipynb

Subsampled 2100 Toronto geolocations, with which fetched 2034 Toronto images. Those images would be used to deploy our final model for the predictions of Perceived Toronto Street View. 

#### CNN model building:

##### Transfer Learning: 

*  ResNeXt50ClassifierTemplate.py

Generates a ResNeXt50 neural network with pretrained weights from Keras.

*  Classifier.py

Trains the models using the pretrained weights, conducts 5-fold cross-validation, and save the 5 best models produced during cross-validation and 1 best model produced during the training on the full training data. 

*  ResNeXt_prediction.py

Generates predictions by each of model on the test set. 

##### Building our own CNN: 

*  Keras-CNN_Tested on Toronto images.ipynb
*  CNN_Tested_Explained_with_Toronto_images.ipynb

Builds a CNN model by ourselves and explains it using LIME.

#### Ensemble:
*  ensemble.ipynb

Conducts ensemble learning methods including: Averaging ensemble; Conditional ensemble; and Weighted ensemble; Calculates each model's confusion matrix and performance;  Finds one final model with relatively best prediction of both safety = 0 and safety =1.
 
#### Model explanation:
*  Keras-CNN_Tested on Toronto images.ipynb
*  CNN_Tested_Explained_with_Toronto_images.ipynb

Builds a CNN model by ourselves and explains it using LIME.


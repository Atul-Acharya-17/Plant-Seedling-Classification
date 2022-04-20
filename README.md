# Plant Seedling Classification
Rank 13/835 Top(1.56%)

## Overview

The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark, has recently released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

The aim of this project is to build a classifier that can identify plant species from an image.  
List of 12 different species:  

* Black-grass 
* Charlock 
* Cleavers 
* Common Chickweed 
* Common wheat 
* Fat Hen
* Loose Silky-bent 
* Maize 
* Scentless Mayweed 
* Shepherds Purse 
* Small-flowered Cranesbill 
* Sugar beet

## Kaggle evaluation

Submissions are evaluated on Mean Score, which at Kaggle is a micro-average F1-score. 

<img src="https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/assets/evaluation.png">

## Approaches

### Data Preprocessing

* Segmentation

    <img src="https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/assets/segmentation.png" width="200" height="200">

    [segmentation code](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/utils/segment_images.py)

* Balancing Dataset

    <img src="https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/assets/augmentation.png" width="200" height="200">

    [augmentation code](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/utils/augment_dataset.py)

### Models and Techniques
* [K-Means Clustering](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/pca_kmeans_knn/pca-knn-kmeans.ipynb)
* [K-Neirest Neighbours Classifier](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/pca_kmeans_knn/pca-knn-kmeans.ipynb)
* [Convolutional Neural Networks](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/cnn/cnn-original-dataset.ipynb)
* [EfficientNetB3](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/EfficientNet/efficientnetb3.ipynb)
* [Xception](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/xception/plantseedling-xception.ipynb)
* [Inception-ResNet-v2](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/inception_resnet_v2/plantseedling-inceptionresnetv2.ipynb)
* [Vision Transformer](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/transformers/vision_transformer_pretrained.ipynb)
* [Weighted Average Ensemble](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/ensemble/ensemble.ipynb)
* [Binary Classifier Error Correction](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/binary_classifier/binary-classifier-inference.ipynb)
* [Test-Time-Augmentation](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/code/test_time_augmentation/plant-seedling-tta.ipynb)

## Final Solution
Our final solution is the weighted average ensemble that combines the probabilities of the Xception, EfficientNetB3 and InceptionResNet-v2 models using a weighted average aggregator

The final probabilities are given by 

P(X) = 0.1 * EfficientNetB3(X) + 0.5 * Inception-ResNet-v2(X) + 0.4 * Xception(X)

<img src="https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/assets/ensemble.png">

## Leaderboard
<img src="https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/assets/leaderboard.png">

## Dataset Links
* [Original Dataset](https://www.kaggle.com/competitions/plant-seedlings-classification/data)
* [Segmented Dataset](https://entuedu-my.sharepoint.com/personal/hienvan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhienvan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FT%C3%A0i%20li%E1%BB%87u%20h%E1%BB%8Dc%20t%E1%BA%ADp%2FCZ4041%2FDatasets%2Ftrain%2Dlarge)
* [Balanced Dataset](https://entuedu-my.sharepoint.com/personal/hienvan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhienvan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FT%C3%A0i%20li%E1%BB%87u%20h%E1%BB%8Dc%20t%E1%BA%ADp%2FCZ4041%2FDatasets%2Ftrain%2Dlarge)
* [Balanced Segmented Dataset](https://entuedu-my.sharepoint.com/personal/hienvan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhienvan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FT%C3%A0i%20li%E1%BB%87u%20h%E1%BB%8Dc%20t%E1%BA%ADp%2FCZ4041%2FDatasets%2Ftrain%2Dlarge%2Dseg)

## Final Submission
[final_submission.csv](https://github.com/Atul-Acharya-17/Plant-Seedling-Classification/blob/master/submissions/final_submission.csv)

## Model Weights Links

* [Model Weights](https://entuedu-my.sharepoint.com/personal/hienvan001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhienvan001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FT%C3%A0i%20li%E1%BB%87u%20h%E1%BB%8Dc%20t%E1%BA%ADp%2FCZ4041%2FModel)

## Team Members

* [Atul Acharya](https://github.com/ABHINAV112)
* [Tran Hien Van](https://github.com/hienvantran)
* [Rachita Agrawal](https://github.com/rachita7)
* [Aks Tayal](https://github.com/tayalaks2001)
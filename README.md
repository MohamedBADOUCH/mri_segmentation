# SEGMENTATION_IRM_CERVEAU

![](https://img.shields.io/badge/Python-31A8FF.svg?logo=python&logoColor=white)
![](https://img.shields.io/badge/Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white)
![](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=google-colab&logoColor=white)
![](https://img.shields.io/badge/scikit%20learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)
![](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Description 

Ce projet a été réalisé dans le cadre du cours de programmation avancée du M2 DS2E. Le but de cette application streamlit est de permettre la détection et la segmentation de tumeurs cérébrales sur toute image stockée en format numpy array.

## Démo

![](data/demo.gif)

## Structure

![Employee data](/data/architecture.png)

## Données

Les données proviennent de [kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) et ont été utilisées pour l'entraînement du modèle. 

## Limites

- L'application doit être lancée depuis google colab (GPU) et un google drive associé : ceci ralentit le programme et empêche sa réutilisation par une autre personne. 
- Le fichier .hdf5 qui contient les paramètres du modèle étant trop lourd, celui-ci ne peut pas être partagé sur github.

## Contributeurs

- Valentin DAAB : [@valentin-daab](https://github.com/valentin-daab)
- Mohamed BADOUCH : [@MohamedBADOUCH](https://github.com/MohamedBADOUCH)
- Osman GULLU : [@croco13](https://github.com/croco13)

## Sources et inspirations

- [Preprocessing, Modèle et Training](https://github.com/bnsreenu/python_for_microscopists/tree/master/231_234_BraTa2020_Unet_segmentation) 
- [Application Streamlit](https://github.com/medicimage/AI_med_seg_app)



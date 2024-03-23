
# Dog / Cat Classifier

A Convolutional Neural Network built using transfer learning with weights from VGG16 for binary classification

## Dataset used: https://www.kaggle.com/datasets/salader/dogs-vs-cats

## Model Architecture
Model takes the top 4 blocks of convolutional layers from VGG16 as it is and finetunes the last block according to the given dataset.

<img width="515" alt="image" src="https://github.com/veer-chheda/catDogClassifierCNN/assets/114056626/7c63485b-7145-441b-a1f2-fe4f24bac7f6">

The model follows a Sequential order with a Flatten layer to reshape output from Convolutional, followed by layers 4 Dense Layers and corresponding Dropouts with 0.2 dropping ratio.

<img width="515" alt="image" src="https://github.com/veer-chheda/catDogClassifierCNN/assets/114056626/087a09cd-6888-4ee0-a4bb-d9ff3459e581">

With a batch size of 32, Data Augmentation is performed to improve the accuracy.

Optimizer used is RMSProp with learning rate as 1e-5. After training for 15 epochs, the model achieves **96.84%** validation accuracy.

![image](https://github.com/veer-chheda/catDogClassifierCNN/assets/114056626/7fac5c75-e5c8-42cd-a1a7-661d9e9f133f)


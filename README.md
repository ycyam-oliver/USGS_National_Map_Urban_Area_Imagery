# USGS_National_Map_Urban_Area_Imagery

There are 21 classes of different land use images in the dataset. All images has been well-labelled by one of the class names. The purpose of this project is to classify the images in the dataset or a subset of the dataset when they are shuffled. In the current codes, 8 out of the 21 classes were taken out to train the model.

The dataset can be downloaded from: http://weegee.vision.ucmerced.edu/datasets/landuse.html

In this project, two CNN models were constructed to classify the images – one is manually constructed and the other is from transfer learning.

Pixels of images from the dataset originally have values from 0 to 255. They are normalized to 0 to 1 for the inputs. Also, the sizes of the images are reduced from 256x256 to 64x64 (or 224x224 in the transfer learning model) in order to focus only on the more important features and reduce the computational resources and time needed to train the models.

(I) Manually constructed CNN model
-------------------------------------------------

In the construction of the CNN model, the number of convolution layers has been tested from two layers and four layers for the best accuracy and least overfitting. Eventually three convolution layers was chosen. 

Throughout the layers of the convolution layers, the number of nodes were increased gradually, in the step of multiples of 2. 

The size of the filters was chosen to be 3x3. In order to keep the importance of the edge of the image, padding=same was used. Max pooling layers were added to keep only the most significant features and reduce the size of data needed to be processed and thus lower the time required for the training.

Activation functions were chosen to be ReLU because it greatly relieves the gradient vanishing problem and has been proven to perform better than other activation functions generally.

To help with the overfitting problem, data augmentation and different methods of regularizations were employed/ attempted. 

In data augmentation, the original input images were augmented dynamically, or ‘on-the’fly’,  with some transformations when they are fed into the model. These transformations include zooming in, height/ width shifting, vertical/ horizontal flip and rotation. This could diversify the inputs to the model during the training and make it more general.

Different regularizations, including dropout, L2 regularization and batch normalization, or a combination of them, have been attempted in the model construction. It was found that using only dropout gives the best accuracy. EarlyStopping and keeping the best model during training were also applied to lower the chance of overfitting.

In the fully connected layers, two hidden layers were used. A dropout layer was added after each hidden layer as mentioned above and this would make the model more robust. 

For the optimizer in training the model, learning rate is an important hyperparameter, in both the efficiency in the training and determining the performance of the model. Methods which are able to adaptively and dynamically adjust the learning rate by momentum in the training can boost both the efficiency of the training and the final accuracy of the model. Given these considerations, ‘Adam’ was used in the model here.

To find the most optimal combination of hyperparameters, Keras Tuner was employed to find the best set of number of nodes in dense layers, dropout rate and base learning rates. 

The accuracy obtained on the training set is 99.7%, the validation set is 96.2% and the test set 95%. 

(II) Transfer learning model
-------------------------------------------------

Since the training set is rather small (only 100 images for each class), using some renowned models well trained on similar image classification problems can much relieve the overfitting problem and lead to a well generalized model in a relatively short training time.

The transfer model used is VGG, which involves consequent convolution layers with different size of filters (e.g. 3x3 to 5x5) to allow more nonlinear effect but involve less parameters to be trained. 

In the training, only the top of the model (the fully connected layer) is set to be trainable to reduce the computational resources needed. Data augmentation was also used for the data input.

Accuracy on the training set is 98.958%, on the validation set is 96.875% and on the test set is 97.5%.

-------------------------------------------------
  Further potential improvements
-------------------------------------------------

1) Try on a subset of dataset with more classes or even all of the 21 classes

2) Try more convolution methods or CNN structures 

3) Try more different transformations in data augmentation

4) Train the whole model in the transfer learning, not just the top of the model



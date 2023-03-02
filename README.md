# CNNs for image classification

Today, deep learning is the state-of-the-art technique used to do image classification. 

In this notebook, we will implement and compare 3 techniques:


*  **First, we will design a simple CNN based on a simple dataset**

We will focus on the famous MNIST dataset and perform digit classification using a simple CNN, presenting basic pre-processing of data and analysis of the results.

*  **Secondly, we will focus on more complex images**

To do so, we will consider [MedMNIST](https://medmnist.com/). It is a large-scale MNIST-like collection of standardized biomedical images, includiing 12 datasets for 2D (images) and 6 datasets for 3D.

*   **Lastly, we will perform transfer learning**

It consists of re-using pre-learned network for another task. Typically, the first part of the CNN is learning a representation of images. We will use this representation in order to classify other classes of images.

To build our neural networks, we will use **Pytorch**, one of the most used libraries for deep learning. While it is a little harder to handle that tensorflow, it allows for easier access to the internal structure of the models, and makes it easier to personalize the model's pipeline.

## Libraries 
1. [Pytorch](https://pytorch.org/docs/stable/index.html):  an optimized tensor library for deep learning using GPUs and CPUs.
2. [Torchvision](https://pytorch.org/vision/stable/index.html): as part of the Pytorch project, it includes popular datasets, model architectures, and common image transformations for Computer Vision.
3. [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/): a collection of 90+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. 
4. [Torchviz](https://github.com/szagoruyko/pytorchviz): a small package to create visualizations of PyTorch execution graphs and traces.
5. [Torchsummary](https://pypi.org/project/torch-summary/): provides information complementary to what is provided by print(your_model) in PyTorch, similar to Tensorflow's model.summary() API to view the visualization of the model.
9. [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html): a package that allows us to display a model's performance in a smart way.
10. [PIL](https://pillow.readthedocs.io/en/stable/): a lbrary that provides extensive file format support, an efficient internal representation, and fairly powerful image processing capabilities.
1. [Tqdm](https://tqdm.github.io/): a simple package to let loops show a smart progress meter.
6. [Numpy](https://numpy.org/doc/1.24/reference/index.html)
7. [Matplotlib](https://matplotlib.org/stable/index.html)


# I. First step : Simple Convolutionnal Neural Network (CNN)

This part of the lab was heavily inspired by the [Pytorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

## Dataset

In this part, we will use the simple and famous [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. This dataset of hand-written digits contains 60000+10000 grey level images of size 28x28 pixels.

![MNIST](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png
)

## Loading and preparing the data

To load the MNIST dataset, we simply import it from `torchvisions.datasets` and then wrap it in a dataloader with a batch size of 64. \
The original notebook explicitly normalized the images, but in our implementation this is not needed as `torchvision.transforms.ToTensor` implements it by default.\
In order to imrove our model's learning, we shuffle the training data on loading by setting `shuffle=True`.

To feed the data to a CNN, we need to shape it as required by Pytorch. As input, a 2D convolutional layer needs a **4D tensor** with shape: **(batch, channels, rows, cols)**. Therefore, we need to precise the "channels" axis, which can be seen as the number of level of color of each input: 3 channels if the image is in RGB, 1 if it is in grayscale.

## Building the model

The model we will use to classify our data is built on the [LeNet](https://en.wikipedia.org/wiki/LeNet) architecture:

<img src="https://drive.google.com/uc?export=view&id=154FJGiTvd7LPUI8JhemeoO8Bl1CG27hw" 
     height="600" />

We build our Pytorch model by creating a custom class called `LeNet` that extends `nn.Module`. \
In it we find:
- `conv_stack`, which implements the convolutional layers of LeNet and to which we will pass our images to extract their features;
- `flatten`, which will flatten our features to be passed to the next stack;
- `dense_stack`, which implements the dense layers of LeNet and to which we will pass the flattened features to classify the images.

Thanks to the the torchviz library, we can check out our model's structure and confirm that it is the same as LeNet's:

<img src="https://github.com/federaspa/CNNs-for-image-classification/blob/main/Images/model_torchviz.png" 
     height="600" />

## Training the model

Our training and validation functions iterate over each batch in the dataloaders and find each batch's output, loss and accuracy, as well as the overall epoch's. \
Everything is logged in a `tensorboard`'s `SummaryWriter`, which we will later use to display our model's performance.

In this part, the model is trained with an AdamW optimizer with 0.001 learning rate, loss is calculated as `CrossEntropyLoss` and the metric we use to assess our model is `MulticlassAccuracy` for 10 classes. We run 20 training epochs.

Note that as Pytorch does not natively implement a Callback function, we implement our own in the training loop by checking at the end of every epoch if the model has made any significant improvement in the past n epochs, where n is defined by `patience`.

## Results

After 20 epochs, our model has reached a satisfying performance on the test set of about . \
We can notice that with our parameters the model does not seem to be overfitting, therefore there is still room to boost it by increasign the number of epochs, tweaking the type and parameters of the optimizer or decreasing the batch size.

# II. Second step: Simple Convolutionnal Neural Network on more complex data

## Dataset

In this part we will be using the PathMNIST dataset from the MedMNIST collection. It consists of 100,000 non-overlapping image patches from histological images, and a test dataset of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task.

<img src="https://medmnist.com/assets/v2/imgs/PathMNIST.jpg" 
     width="300" />
     
## Loading and preparing the data

It is possible to [download the PathMNIST dataset](https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb) in a similar fashion to MNIST, but since it is quite substantial, this time we import it from local files.\
Specifically, we have 6 `.npy` files, containing the images and labels for the training, validation and test datasets.\
In order to load them, we create a custom Pytorh Dataset class called `MedMNISTDataset`, whose `__getitem__` will iterate through all the files in the respective folders and pair up each image with its corresponding label. We then load the datasets into dataloaders with `batch_size = 64`, shuffling the training data.

## Building and training the model

### Architecture
The model's architecture for this part is almost identical to the one we used to classify the MNIST dataset. The only difference is in the last Linear layer, which outputs 9 classes instead of 10:

<img src="https://github.com/federaspa/CNNs-for-image-classification/blob/main/Images/model_med_torchviz.png" 
     width="600" />

### Training
The model's training is also almost identical to the one for the first part, with the only difference being the number of classes we pass to the MultiClassAccuracy metric.

# III. Third step: Transfer Learning

Transfer Learning is the reuse of an already trained model for another task. There are many reasons we could decide to do it, ranging from lack of resources to small datasets.
In our case, since we will try to fine-tune the pre-trained `VGG16` model to work on our dataset. We will try two approaches:
* First, we will freeze the model's layers and add our own classifier to the feature-extraction stack.
![fine_tuning_freeze](https://drive.google.com/uc?export=view&id=1kBfY0Jssj0EcAOt0HYdLyj4UjTe3wKaQ)
* Then, we will try to fine-tune the last convolutional layer of `VGG16` to adapt it to our own dataset.
![fine_tuning_unfreeze](https://drive.google.com/uc?export=view&id=1GmWHBWqs3f8WZoYDZYx_uBHNf-_gocfN)

## Dataset

For this last part, we will use the [Animal-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset, consisting of about 28K medium quality animal images belonging to 10 categories.

![Cane](https://storage.googleapis.com/kagglesdsdata/datasets/59760/840806/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230302%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230302T104722Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=56cdcfd09ac46d923f3ff3e83d9210b0096f5d1df043f48533b069efde6d352ff8c5d91f496d9113804c1fe353567d0fbb36d08d27b17465a60989ddd12f884df4d0f1ef112429b1103248e5d86362c1f2c9754a57a3d0e4ad7d74ff0dd57e63d86c63e965320159356fb004c8d89a0d88d2ee6f9e78acd726cb980f22143329eec75363f919c10b8369798fa1642ad7e7511b9431f817421e29655a55f94939295e005fbdc8ec27985819ea18b81977638fbeb1080776f257165b0a1c2c4c602693d077bc7f96f5392ff4c3f93e2153a5e8d37d5cb70ea93e12f16ed24c7df536f87deff8917efd7ec3a24f8fd997bd36c940f58eb434dca52f5c16dc77eb68)

## Loading the data

As for the previous part, the dataset is quite big, therefore we load it from a local directory. We first create, for both the training and the test split, two numpy arrays containing the images and labels respectively. Then we use a custom Dataset called `AnimalDataset` to wrap the images in a dataset like we did in the previous part.\
Note that we reshape the images from `(N,H,W,C)` to `(N,C,H,W)` in order to feed them to `VGG16`.\
Since this dataset does not come with a validation split, we will create the validation split ourselves.\
Note that after the images are loaded, they look slightly corrupted:\
![Cow](https://github.com/federaspa/CNNs-for-image-classification/blob/main/Images/cow.png)

This is normal and is an effect of our resizing and conversion steps we implemented at loading.

## Building and Training the model

### Architecture

Creating the feature extraction part of our model is only a matter of importing it from the `torchvision` library.\
In doing so, however, we also import the classifier part, that we will replace with our own, which consists of:
* One linear layer with 1024 neurons and a ReLu activation function;
* One linear layer with 2 neurons and a Softmax activation function.

We also start by freezing all the network's layers except for our classifier's.

As the model architecture is quite big, we only display the classifier's:

![animal_model](https://github.com/federaspa/CNNs-for-image-classification/blob/main/Images/model_animal_torchviz.png)

### Training

The model's training is almost identical to the ones before, again with the only differences being:
* The number of classes we pass to the MultiClassAccuracy metric;
* We use RMSprop with 0.01 learning rate instead of AdamW with 0.001 learning rate

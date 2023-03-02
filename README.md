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

# I. First step : Simple Convolutionnal Neural Network (CNN)

This part of the lab was heavily inspired by the [Pytorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

## Libraries 
1. [Pytorch](https://pytorch.org/docs/stable/index.html):  an optimized tensor library for deep learning using GPUs and CPUs.
2. [Torchvision](https://pytorch.org/vision/stable/index.html): as part of the Pytorch project, it includes popular datasets, model architectures, and common image transformations for Computer Vision.
3. [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/): a collection of 90+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. 
4. [Torchviz](https://github.com/szagoruyko/pytorchviz): a small package to create visualizations of PyTorch execution graphs and traces.
5. [Torchsummary](https://pypi.org/project/torch-summary/): provides information complementary to what is provided by print(your_model) in PyTorch, similar to Tensorflow's model.summary() API to view the visualization of the model.
9. [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html): a package that allows us to display a model's performance in a smart way.
1. [Tqdm](https://tqdm.github.io/): a simple package to let loops show a smart progress meter.
6. [Numpy](https://numpy.org/doc/1.24/reference/index.html)
7. [Matplotlib](https://matplotlib.org/stable/index.html)

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

We build our Pytorch model by creating a custom class called `LeNet` that extends `nn.Module`. \
In it we find:
- `conv_stack`, which implements the convolutional layers of LeNet and to which we will pass our images to extract their features;
- `flatten`, which will flatten our features to be passed to the next stack;
- `dense_stack`, which implements the dense layers of LeNet and to which we will pass the flattened features to classify the images.

The model we will use to classify our data is built on the [LeNet](https://en.wikipedia.org/wiki/LeNet) architecture:

<img src="https://drive.google.com/uc?export=view&id=154FJGiTvd7LPUI8JhemeoO8Bl1CG27hw" 
     height="600" />

Thanks to the the torchviz library, we can check out our model's structure and confirm that it is the same as LeNet's:

<img src="https://github.com/federaspa/CNNs-for-image-classification/blob/main/Images/model_torchviz.png" 
     height="600" />

## Training the model

Our training and validation functions iterate over each batch in the dataloaders and find each batch's output, loss and accuracy, as well as the overall epoch's. \
Everything is logged in a `tensorboard`'s `SummaryWriter`, which we will later use to display our model's performance.

In this part, the model is trained with an AdamW optimizer with 0.001 learning rate, loss is calculated as `CrossEntropyLoss` and the metric we use to assess our model is `MulticlassAccuracy` for 10 classes .

Note that as Pytorch does not natively implement a Callback function, we implement our own in the training loop by checking at the end of every epoch if the model has made any significant improvement in the past n epochs, where n is defined by `patience`.

# II. Second step: Simple Convolutionnal Neural Network on more complex data

## Dataset

In this part we will be using the PathMNIST dataset from the MedMNIST collection. It consists of 100,000 non-overlapping image patches from histological images, and a test dataset of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task.

<img src="https://medmnist.com/assets/v2/imgs/PathMNIST.jpg" 
     width="300" />

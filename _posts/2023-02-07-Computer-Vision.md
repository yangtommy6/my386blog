---
layout: post
title: "Deep leaning with Fast.ai"
author: Christian Yang
description: Using Fast.ai Library to Build a Cat and Dog Image Recognition App
image:
---

## Using Fast.ai Library to Build a Cat and Dog Image Recognition App

In recent years, deep learning has become the leading method for image recognition and classification. With the rise of deep learning, it is now possible to train computers to recognize images with great accuracy. In this blog post, we will show you how to build a simple image recognition app that can tell if an image contains a cat or a dog using the Fast.ai library.

## Introducing Fast.ai

You might think that deep learning sounds like something extremely hard to learn and understand, but Jeremy Howard makes it approachable to eveyone. Fast.ai is an open-source deep learning library that makes it easy to build and train deep learning models. The library is built on top of PyTorch and provides a high-level API for training and evaluating deep learning models. In this tutorial, we will be using Fast.ai to build a convolutional neural network (CNN) that can recognize cats and dogs in images.

## Prerequisites

To follow along with this tutorial, you will need the following:

1. Basic knowledge of Python programming
2. A computer with Python and Jupyter Notebook installed, you can also use Colab or Kaggle.

Here is the full code:
https://colab.research.google.com/drive/1REmEqpWZw3nTDEYCmuUNsUwMglNdbwnG?usp=sharing

The steps below will help you understand how everything works. If you want to learn more, go to fast.ai and they have an entire free practical deep learning class for everyone!

## Step 1: Importing the Required Libraries

The first step in building the app is to import the required libraries. For this project, we will be using the Fast.ai library and the following libraries:

```
from fastbook import *
from fastai.vision.widgets import *
```

## Step 2: Loading the Dataset, Labeling the images, and data argumentation

For this tutorial, we will be using a dataset of cat and dog images that can be easily downloaded from the internet. After downloading the dataset, you should have a folder with two subfolders - one for cats and one for dogs. Each subfolder should contain several images of cats or dogs.

How human learn to seperate cats and dogs? We have our mom pointing a dog and tell us: "This is a dog.", our brain then connect what we see(a dog) and the word "dog". We have collected the pictures of dogs and cats from the previous steps, now we will need to label them correctly for the computer.

So, we need to write a function that will load the images from the dataset and convert them into a format that can be used by the model. The function will also convert the labels (cat or dog) into one-hot encoded vectors.

```
key = os.environ.get('AZURE_SEARCH_KEY', '90086a15a7704edeb954308d3595f8a1')#Using Azure search to get the pictures. To do that, we need to hav the azure search key.
search_images_bing
animal_types = 'cat', 'dog'
path = Path('animals')
```

```
animals = DataBlock(
    blocks=(ImageBlock, CategoryBlock), #Image is the input, categoryblock is the label
    get_items=get_image_files, #grab the input
    splitter=RandomSplitter(valid_pct=0.2, seed=42), #Validation set, seed can set validation to certain pics
    get_y = parent_label,#grab the label, parent_label is the path
    item_tfms=Resize(128))
```

```
#Data Augmentation
animals = animals.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = animals.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

## Step 3: Building the Model

Now that we have loaded the dataset, we can build the model.

```
#Training the model
animals = animals.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = animals.dataloaders(path)
```

```
learn = vision_learner(dls, resnet18, metrics=error_rate) #dls--data loder, resent18--model, metrics: how you calculate the accuracy
learn.fine_tune(4) #the function to tell the program start to train the model, number can decide how many times to learn
```

## Step 4:Export the trained model

```
learn.export('model.pkl')
```

Now we got our trained model, we can build a app or deploy it online to use it. I used HuggingFace as the online platform to share this app.

You can see the result of this app at: https://huggingface.co/spaces/yangtommy6/Animal_Project
Try to put different cat or dog's picture and see how the program is able to reconize the image!

## Conclusion

Fast.ai's deep learning course is very practical, although I didn't have any deep learning experience before I started the course, I was able to learn the concepts of deep learning, and implement them with the fast.ai library. I actually got an cybersecurity analyst internship partially because of having a basic concept of computer vision and deep learning. I believe that deep learning is a must have skill for a data science career.

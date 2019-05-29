import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import json
import PIL
from PIL import Image
import argparse

import functioncaller

#Command Line Arguments

argument = argparse.ArgumentParser(description='predict-file')
argument.add_argument('--input_image', default='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg', type = str)
argument.add_argument('--checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', action="store",type = str)
argument.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
argument.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
argument.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = argument.parse_args()
image_path = args.input_image
number_of_outputs = args.top_k
path = args.checkpoint
power = args.gpu
jsonfile = args.category_names
 

training_loader, testing_loader, validation_loader, train_data = functioncaller.load_transform_data()

model = functioncaller.load_checkpoint(path)

with open('./ImageClassifier/' + jsonfile, 'r') as json_file:
    cat_to_name = json.load(json_file)

probabilities = functioncaller.predict(image_path, model, number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Completed")

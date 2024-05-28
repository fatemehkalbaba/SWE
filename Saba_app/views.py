from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, 4, 2)
        self.conv2 = nn.Conv2d(64, 192, 5, 1, 2)
        self.conv3 = nn.Conv2d(192, 424, 3, 1, 1)
        self.fc1 = nn.Linear(424*8*8, 200)
        self.fc2 = nn.Linear(200, 160)
        self.fc3 = nn.Linear(160, 100)
        self.fc4 = nn.Linear(100, 2)
        self.drop = nn.Dropout2d(p=0.3)
        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 424*8*8)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.drop(X)
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

model = CNN()
model.load_state_dict(torch.load("models/tumor_classification.pt"))
def Home(request):
    return render(request, "home_page.html", {})


def Predict(request):
    print(request)
    print(request.POST.dict())
    fileObj = request.FILES["filePath"]
    print(fileObj)
    fs = FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)

    test_image = Image.open("."+filePathName).convert("RGB")

    test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((290)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.33], std=[0.229, 0.224, 0.225])
    ])
    test_image = test_transform(test_image)
    test_image = test_image.view(1, 3, 290, 290)
    model.eval()
    with torch.no_grad():
        pred = model(test_image).argmax()

    if pred.item() == 0:
        pred = "no"
    elif pred.item() == 1:
        pred = "yes"
    print(pred)



    return render(request, "home_page.html", {"filePathName":filePathName, "predictedLable":pred})

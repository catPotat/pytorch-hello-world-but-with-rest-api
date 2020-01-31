from rest_framework import (
    permissions,
    status,
)
from rest_framework.views import APIView
from rest_framework.response import Response
# from rest_framework.parsers import 

import io, os
from pathlib import Path
from PIL import Image

import torch
from torchvision import models
import torchvision.transforms as transforms
from ml_model.model import Hewwo


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        # transforms.Resize(28*28),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.485, 0.456, 0.406],
        #     [0.229, 0.224, 0.225]
        # )
    ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((28,28))
    return my_transforms(image).mean(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)

    PATH = os.path.abspath('.')
    net = Hewwo()
    net.load_state_dict(torch.load(PATH+"\\ml_model\\hewwo.pth"))

    # import matplotlib.pyplot as plt
    # plt.imshow(tensor, cmap='gray')
    # plt.show()

    output = net.forward(tensor.view(-1, 28*28))
    return torch.argmax(output)



class HandPridictionAPIView(APIView):

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        img_bytes = file_obj.read()
        return Response(
            {"this": get_prediction(img_bytes)},
            status = status.HTTP_200_OK
        )

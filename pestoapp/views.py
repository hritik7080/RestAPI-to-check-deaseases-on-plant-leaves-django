from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from keras.models import load_model
from keras.preprocessing import image
from .models import File
import numpy as np
import os


def predict_disease(image_path):
    classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy',
               'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight',
               'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    new_img = image.load_img(f'.{image_path}', target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    model = load_model('./model.hdf5')
    pred = model.predict(img)
    d = pred.flatten()
    j = d.max()
    for index, item in enumerate(d):
        if item == j:
            class_name = classes[index]

    return [image_path, class_name, str(j)]


class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            file_serializer = FileSerializer(data=request.data)
            if file_serializer.is_valid():
                file_serializer.save()
                output = predict_disease(dict(file_serializer.data)['file'])
                objs = File.objects.all()
                objs = [i for i in objs]
                objs[-1].delete()
                print(dict(file_serializer.data)['file'])
                os.remove(f".{dict(file_serializer.data)['file']}")
                return Response({"filename": output[0], "disease": output[1], "confidence": output[2]},
                                status=status.HTTP_201_CREATED)
            else:
                return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except:
            objs = File.objects.all()
            objs = [i for i in objs]
            objs[-1].delete()
            os.remove(f".{dict(file_serializer.data)['file']}")
            return Response({"error": "an error occurred"}, status=status.HTTP_400_BAD_REQUEST)

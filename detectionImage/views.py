from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
import cv2
import numpy as np

from detectionImage.models import FaceDetection
from .serializers import ImageUploadSerializer

def detect_face(image):
    np_image = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    save_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_path = os.path.join(save_path, 'uploaded_image.jpg')
    cv2.imwrite(image_path, img)
    
    
     
    detected_name = "John Doe"
    return image_path, detected_name

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            image_path, detected_name = detect_face(image)
            
            # Process data 
            
            image_url = request.build_absolute_uri(
                os.path.join(settings.MEDIA_URL, 'uploaded_images', 'uploaded_image.jpg')
            )
            response_data = {}
            # response_data = {
            #     [
            #         {
            #             "image_url": image_url,
            #             "name": detected_name
            #         }
            #     ]
               
            # }
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def face_detection_list(request):
    face_detections = FaceDetection.objects.all()
    return render(request, 'face_detection_list.html', {'face_detections': face_detections})
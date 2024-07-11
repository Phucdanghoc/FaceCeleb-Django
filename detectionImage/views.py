from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
import cv2
import numpy as np
from django.core.paginator import Paginator

from detectionImage.models import FaceDetection
from .serializers import ImageUploadSerializer

def getFaceDetectByIndex(indexs):
    return FaceDetection.objects.filter(index__in=indexs)

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            image_path, detected_name = detect_face(image)
            image_url = request.build_absolute_uri(
                os.path.join(settings.MEDIA_URL, 'uploaded_images', 'uploaded_image.jpg')
            )
            
            # search by list index ... example : ([1,2,3])
            face_detections = getFaceDetectByIndex([20,1232,2932])
            
            response_data = [
                {
                    'index': fd.index,
                    'label': fd.label,
                    'url_image': fd.url_image,
                    'url_npy': fd.url_npy
                }
                for fd in face_detections
            ]
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




def face_detection_list(request):
    face_detections = FaceDetection.objects.all()
    paginator = Paginator(face_detections, 10)  # Show 10 face detections per page

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'face_detection_list.html', {'page_obj': page_obj})

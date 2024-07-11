from django.urls import path
from .views import ImageUploadView
from detectionImage import views

urlpatterns = [
    path('upload', ImageUploadView.as_view(), name='image-upload'),
    path('face-detections/', views.face_detection_list, name='face_detection_list'),

]

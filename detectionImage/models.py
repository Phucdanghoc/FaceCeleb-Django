# myapp/models.py

from django.db import models

class FaceDetection(models.Model):
    index = models.IntegerField()
    label = models.TextField()
    url_image = models.TextField()
    url_npy = models.TextField()

    def __str__(self):
        return f'{self.label} - {self.index}'

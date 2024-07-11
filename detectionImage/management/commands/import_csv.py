import csv
from django.core.management.base import BaseCommand
from detectionImage.models import FaceDetection

class Command(BaseCommand):
    help = 'Import face detection data from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The CSV file to import')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']

        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
        
            for row in reader:
                print(row)
                FaceDetection.objects.create(
                    index=int(row['index']),  
                    label=row['label'],
                    url_image=row['url_image'],
                    url_npy=row['url_npy']
                )
        self.stdout.write(self.style.SUCCESS('Successfully imported data'))
# "D:\DACNTT\all\all\out.csv"
# index,features,label,url_image,url_npy

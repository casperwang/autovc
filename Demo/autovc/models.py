from django.db import models

# Create your models here.

class User(models.Model):
    
    user_name = models.CharField(max_length=30)
    
    user_wav = models.FileField(upload_to='wav/')
    
    def __str__(self):
        return self.user_name
from django.db import models

class Sudoku(models.Model):
   photo =models.ImageField(verbose_name='Imagen', null=True, blank=True,upload_to='main/picture/')
   target=models.CharField(verbose_name='target', null=True, blank=True,max_length=81)
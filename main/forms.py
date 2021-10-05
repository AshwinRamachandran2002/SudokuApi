from django import forms
from django.db.models.base import Model
from django.forms import ModelForm
from .models import Sudoku

class UploadFileForm(ModelForm):
    class Meta:
        model = Sudoku
        fields = ['photo','target']  

    def __init__(self, *args, **kwargs):
        super(UploadFileForm, self).__init__(*args, **kwargs)
        self.fields['photo'].required = False      
        self.fields['target'].required = False      
  

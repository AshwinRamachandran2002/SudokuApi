from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import UploadFileForm
from .processing import process_and_predict
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def solve(request):
    print(request)
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            print(request.FILES)
            print(request.FILES['photo'].name)
            form.save()
            
            digit= process_and_predict(request.FILES['photo'].name)
            return digit
        else:
            print(form.errors)
    else:
        return 404,"not a post request"
    #return render(request,'solve.html',{'form':form})


def retrain(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            retrain(request.FILES['photo'].name,form.target)
            return "success"
        else:
            print(form.errors)
    else:
        return 404,"not a post request"
    #return render(request,'solve.html',{'form':form})


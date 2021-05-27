from django.shortcuts import render
def predict(request):
    imgData = request.POST.get('img')
    convertImage(imgData)
    x = imread(OUTPUT, mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return JsonResponse({"output": response})


def index(request):
    return render(request, 'index.html', {})
# urls.py
from django.urls import path
from .views import index
urlpatterns = [
    path('', index, name="index")
]
from django.shortcuts import render
def predict(request):
    # model=tf.keras.models.load_model("./anoth_model11.h5")
    # result=0
    # imgData = cv2.imread('./img.png')
    # img = np.asarray(imgData)
    # img=img.reshape(1,28,28,1)
    # predictions = model.predict(img)
    # classIndex = model.predict_classes(img)
    # probabilityValue = np.amax(predictions)
    
    # if probabilityValue > 0.3:
    #     result=(classIndex[0])
    # else:
    #     result=(0)
    result=0
    return (result)
def index(request):
    digit=predict(request)
    digit=0
    return render(request, 'index.html', {'digit':digit})
# urls.py
from django.urls import path
from .views import index
urlpatterns = [
    path('', index, name="index")
]
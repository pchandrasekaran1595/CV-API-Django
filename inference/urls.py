from django.urls import path
from .views import classify, detect, segment


app_name = "inference"


urlpatterns = [
    path('classify/', view=classify, name="classify"),
    path('detect/', view=detect, name="detect"),
    path('segment/', view=segment, name="segment"),
]
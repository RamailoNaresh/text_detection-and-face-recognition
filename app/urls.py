from django.urls import path
from app import views


urlpatterns = [
    path("", views.text_detection_google_cloud_vision, name='analyze image'),
    path("compare_face/", views.face_compare, name= 'compare face')
]

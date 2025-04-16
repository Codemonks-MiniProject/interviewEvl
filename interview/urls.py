from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('interview/', views.interview_page, name='interview_page'),
    path('upload/', views.upload_video, name='upload_video'),
    path('result/', views.result_page, name='result_page'),
]
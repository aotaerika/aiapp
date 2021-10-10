from django.urls import path
from . import views



urlpatterns=[
    path("",views.index, name="home"),
    path("",views.tf_idf, name="tf_idf")
    
    
]



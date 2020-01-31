from django.urls import path

from . import views

urlpatterns = (
    path('predict', views.HandPridictionAPIView.as_view(), name='hand-written-pridiction'),
)
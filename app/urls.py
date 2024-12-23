from django.urls import path
from app import views

urlpatterns = [
    path('video_feed/', views.video, name='video'),
    path('video_feed/violator-image/', views.violator_image, name='violator_image'),
    path('video_feed/record/', views.violator_list, name='violator_list'),
    path('video_feed/quiz/', views.quiz_view, name='quiz'),
    path('video_feed/violator-report/', views.violator_report, name='violator_report'),
    path('export-to-excel/', views.export_to_excel, name='export_to_excel'),
    #path('quiz/success/', views.quiz_success, name='quiz_success'), 
    # path('video_feed/quiz/success/<int:score>/', views.quiz_success, name='quiz_success'),
    path('video_feed/select-compliances/', views.select_compliances, name='select_compliances'),]
    


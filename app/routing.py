# your_app_name/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/video_feed/(?P<video_id>\w+)/$', consumers.VideoFeedConsumer.as_asgi()),
    re_path('ws/upload_video/', consumers.UploadVideo.as_asgi()),
]

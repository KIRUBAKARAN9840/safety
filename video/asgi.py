import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'video.settings')  # Replace 'your_project_name' with the actual name of your project

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from app import routing  # Import the routing module containing WebSocket URLs

application = ProtocolTypeRouter({
    "http": get_asgi_application(),  # Handles HTTP connections
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns
        )
    ),
})




# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'video.settings')

# application = get_asgi_application()






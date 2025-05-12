from django.urls import path
from core.views import *

urlpatterns = [
    path('', home, name='home'),  # This is your homepage
    path('accounts/Signup/', signup_view, name='signup'),
]

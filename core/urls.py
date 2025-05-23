from django.urls import path
from core.views import *
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', home, name='home'),  # This is your homepage
    
    # Authentication routes
    path('accounts/signup/', signup_view, name='signup_view'),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),

# Prediction page
    path('predict/', predict_view, name='predict'),

# Chatbot API endpoint
    path('chatbot-api/', chatbot_api, name='chatbot_api'),

]

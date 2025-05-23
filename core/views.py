# views.py

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import numpy as np
import joblib
import os
import yfinance as yf
import torch
import torch.nn as nn

# Define your LSTMModel class exactly as in training
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Paths to model and scaler files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = r"C:\Users\manis\Downloads\stock_lstm_model_pytorch.pth"
SCALER_PATH = r"C:\Users\manis\Downloads\scaler_pytorch.pkl"

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def home(request):
    return render(request, 'index.html')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/accounts/login/')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

@login_required
def predict_view(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol', '').upper()
        try:
            df = yf.download(symbol, period='120d')
            if df.empty or 'Close' not in df.columns:
                raise ValueError("Failed to retrieve stock data.")

            data = df['Close'].values[-60:]
            if len(data) < 60:
                raise ValueError("Not enough data for prediction.")

            scaled_data = scaler.transform(data.reshape(-1, 1))
            X_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, 60, 1)

            with torch.no_grad():
                prediction = model(X_input)
            
            predicted_price = scaler.inverse_transform(prediction.cpu().numpy())[0][0]

            context = {
                'symbol': symbol,
                'predicted_price': round(predicted_price, 2),
                'date': df.index[-1].strftime("%Y-%m-%d")
            }
            return render(request, 'prediction_result.html', context)
        except Exception as e:
            return render(request, 'prediction_result.html', {'error': str(e)})
    return render(request, 'predict_form.html')

@login_required
def chatbot_api(request):
    if request.method == 'POST':
        user_msg = request.POST.get('message')
        # Placeholder response for now
        response = "This is a placeholder AI response. Chatbot integration coming soon!"
        return JsonResponse({'response': response})

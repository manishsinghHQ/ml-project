from django.shortcuts import render,redirect




def home(request):
    return render(request, 'index.html')  # Render the HTML file


def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/accounts/login/')  # Redirect after successful signup
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})


# Create your views here.

from django.shortcuts import render
from .models import User 
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
@csrf_exempt

# Create your views here.

def add(request):
    if request.method == "POST":
        user_name = request.POST.get('user_name')
        user_wav = request.FILES.get('user_wav')
        user = User(user_name=user_name, user_wav=user_wav)
        user.save()
        return render(request, 'add.html', locals())
    return render(request, 'add.html', locals())
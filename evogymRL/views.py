from django.shortcuts import render
from .models import Reward
from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions
from django.http import HttpResponse
# Create your views here.

@api_view(["POST"])
@permission_classes((permissions.AllowAny,))
def get_reward_params(req):
    rew = Reward()
    print(rew.get_func_choices())
    print(rew.get_oper_choices())
    return HttpResponse("data sent")
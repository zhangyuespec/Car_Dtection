from django.shortcuts import render, HttpResponse,redirect
from django.http import JsonResponse
from app1 import models
from app1 import forms
from django.contrib import auth


# Create your views here.

def login(request):
    ret={"msg":"","status":0}
    if request.method=="POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        is_user = auth.authenticate(username=username,password=password)
        if is_user:
            auth.login(request,is_user)
            ret["msg"]="/index/"
        else:
            ret["status"]=1
            ret["msg"]="用户名或者密码错误"
        return JsonResponse(ret)

    return render(request,"login.html")


def register(request):
    form_obj = forms.RegForm()
    if request.method == "POST":
        ret = {"status": 0, "msg": ""}
        form_obj = forms.RegForm(request.POST)
        # print(form_obj.cleaned_data)
        if form_obj.is_valid():
            form_obj.cleaned_data.pop("re_password")
            avatar_image = request.FILES.get("avatar")
            print(form_obj.cleaned_data)
            print(avatar_image)
            models.UserInfo.objects.create_user(**form_obj.cleaned_data, avatar=avatar_image)
            ret["msg"] = "/login/"  # 注册成功跳转到首页
            return JsonResponse(ret)
        else:
            print("不是post")
            ret["status"] = 1
            ret["msg"] = form_obj.errors
            return JsonResponse(ret)
    return render(request, "register.html", {"form_obj": form_obj})


def check_username_exist(request):
    ret = {"status": 0, "msg": ""}
    username = request.GET.get("u")
    print(username,123)
    is_exist = models.UserInfo.objects.filter(username=username)
    if is_exist:
        ret["status"] = 1
        ret["msg"] = "用户名已被注册"

    return JsonResponse(ret)

def index(request):
    return render(request,"index.html")

def test(request):
    return render(request,"test.html")

def logout(request):
    username = request.user.username
    auth.logout(request)
    return redirect("/index/")
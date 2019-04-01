from django.db import models

# Create your models here.

from django.contrib.auth.models import AbstractUser


class UserInfo(AbstractUser):
    """
    用户信息表
    """
    nid = models.AutoField(primary_key=True)
    phone = models.CharField(max_length=11, null=True, unique=True)
    avatar = models.FileField(upload_to="avatars/", default="avatars/default.png", verbose_name="头像")
    create_time = models.DateTimeField(auto_now_add=True)
    is_oursuperuser = models.BooleanField(default=False)# 判断是否是超级用户



    def __str__(self):
        return self.username

    class Meta:
        verbose_name = "用户信息"
        verbose_name_plural = verbose_name
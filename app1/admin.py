from django.contrib import admin
from .models import *
# Register your models here.
from django.utils.safestring import mark_safe

class UserInfoConfig(admin.ModelAdmin):
    def delets(self,):
        return mark_safe("<a href="">删除</a>")

    list_display = ["username","is_oursuperuser",delets]
    # list_display_links = ["is_oursuperuser"]
    list_filter = ["is_oursuperuser","username"]
    list_editable = ["is_oursuperuser"]
    search_fields = ["username"]

    def patch_init(self,request,queryset):
        queryset.update(is_oursuperuser=False)

    patch_init.short_description = "批量初始化"
    actions = [patch_init,]


admin.site.register(UserInfo,UserInfoConfig)
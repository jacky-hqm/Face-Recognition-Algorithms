import os
import platform


# 工程目录
def project_dir():
    return os.path.dirname(os.path.abspath(__file__))


# 支持的图片格式
def img_format():
    img_format = ['.jpg', 'jpeg', '.png', '.bmp']
    return img_format


def platform_judge():
    # Linux 返回True  ,Windows 返回 False
    is_Linux = False
    if 'Windows' in platform.system():
        is_Linux = False
    elif 'Linux' in platform.system():
        is_Linux = True
    return is_Linux


def get_separator():
    if platform_judge():
        separator = '/'
    else:
        separator = '\\'
    return separator

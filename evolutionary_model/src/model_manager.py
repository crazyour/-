import os
from enum import Enum
import torch
import joblib
import tensorflow as tf
import shutil


    
    
def clear_model(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            os.unlink(file_path)

def copy_model(from_path, to_path):
    for filename in os.listdir(from_path):
        src_file = os.path.join(from_path, filename)
        dest_file = os.path.join(to_path, filename)
        shutil.copy2(src_file, dest_file)


"""用户需要自定义的函数"""
def load_model(model_path):
    """
    用户修改次处代码，
    将模型加载入cpu/gpu
    """
def get_model_parameters(model):
    """
    用户修改次处代码，
    输入为加载后的模型
    输出为模型的参数字典
    """
   #return parameters
def save_model(parameters, model_path):
    """
    用户修改次处代码，
    初始化一个无参数的模型，
    然后把parameters加载进去，
    并保存到model_path
    """

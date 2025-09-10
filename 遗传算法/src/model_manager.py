import os
from enum import Enum
import torch
import joblib
import tensorflow as tf
import shutil
from stable_baselines3 import DQN
from src.DQN import create_model
    
    
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

def load_model(model_params_path, env=None):
    pass

def get_model_parameters(model_params_path):
   # 获取模型的参数
   return torch.load(model_params_path)

def save_model(parameters, save_model_params_path):
    # 保存模型参数
    torch.save(parameters, save_model_params_path)
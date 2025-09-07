import os
from enum import Enum
import torch
import joblib
import tensorflow as tf
import shutil

class ModelType(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    CUSTOM = "custom"

def detect_model_type(path: str) -> ModelType:
    """根据文件扩展名自动识别模型类型"""
    if path.endswith(".pt") or path.endswith(".pth"):
        return ModelType.PYTORCH
    elif path.endswith(".h5") or os.path.isdir(path) and "saved_model.pb" in os.listdir(path):
        return ModelType.TENSORFLOW
    elif path.endswith(".pkl") or path.endswith(".joblib"):
        return ModelType.SKLEARN
    else:
        return ModelType.CUSTOM

def load_model(path: str):
    """统一的模型加载接口，自动检测模型类型"""
    model_type = detect_model_type(path)
    
    if model_type == ModelType.PYTORCH:
        return load_pytorch_model(path)
    elif model_type == ModelType.TENSORFLOW:
        return load_tensorflow_model(path)
    elif model_type == ModelType.SKLEARN:
        return load_sklearn_model(path)
    elif model_type == ModelType.CUSTOM:
        raise NotImplementedError("请为自定义模型实现加载逻辑")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def load_pytorch_model(path: str):
    """加载 PyTorch 模型"""
    try:
        model = torch.load(path, map_location=torch.device('cpu'))
        return model
    except Exception as e:
        raise RuntimeError(f"加载 PyTorch 模型失败: {e}")

def load_tensorflow_model(path: str):
    """加载 TensorFlow/Keras 模型"""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        raise RuntimeError(f"加载 TensorFlow 模型失败: {e}")

def load_sklearn_model(path: str):
    """加载 Scikit-learn 模型"""
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"加载 Scikit-learn 模型失败: {e}")

def save_model(model, params, path):
    """统一的模型保存接口，自动检测模型类型"""
    model_type = detect_model_type(path)
    
    if model_type == ModelType.PYTORCH:
        # 更新 PyTorch 模型的参数
        model.load_state_dict(params)  # 将参数加载到模型中
        # 保存完整的模型（结构 + 参数）
        torch.save(model, path)
    elif model_type == ModelType.TENSORFLOW:
        # 更新 TensorFlow/Keras 模型的参数
        model.set_weights([params[name] for name in params.keys()])
        # 保存完整的模型
        model.save(path)
    elif model_type == ModelType.SKLEARN:
        # Scikit-learn 模型不需要单独更新参数，直接保存整个模型
        joblib.dump(model, path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
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

def get_model_parameters(model):
    """
    输入一个模型，输出其参数字典。
    支持 PyTorch、TensorFlow/Keras、Scikit-learn。
    """
    # PyTorch
    if hasattr(model, 'named_parameters'):
        return {name: param.clone() for name, param in model.named_parameters()}
    # TensorFlow/Keras
    elif hasattr(model, 'get_weights') and hasattr(model, 'weights'):
        return {w.name: w.numpy().copy() for w in model.weights}
    # Scikit-learn
    elif hasattr(model, 'get_params'):
        return model.get_params(deep=True)
    else:
        raise ValueError("不支持的模型类型或无法提取参数")


import os

def rename_models():
    """
    将指定目录中的模型文件重命名为统一格式，例如 0, 1, 2 等（保留原始后缀）。
    """
    directory = "models/parents"  # 模型文件所在的固定目录
    for idx, filename in enumerate(os.listdir(directory)):
        old_path = os.path.join(directory, filename)
        # 保留原始文件后缀
        extension = os.path.splitext(filename)[1]
        new_path = os.path.join(directory, f"{idx}{extension}")
        os.rename(old_path, new_path)
        print(f"重命名: {old_path} -> {new_path}")
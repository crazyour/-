import os

def rename_models():
    """
    将指定目录中的模型文件重命名为统一格式，例如 0, 1, 2 等（保留原始后缀），并返回模型数量和后缀名。
    """
    directory = "models/parents"  # 模型文件所在的固定目录
    count = 0
    extension = None
    for idx, filename in enumerate(os.listdir(directory)):
        old_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1]
        if extension is None:
            extension = ext  # 记录第一个文件的后缀
        new_path = os.path.join(directory, f"{idx}{ext}")
        os.rename(old_path, new_path)
        print(f"重命名: {old_path} -> {new_path}")
        count += 1
    return count, extension  # 返回数量和后缀
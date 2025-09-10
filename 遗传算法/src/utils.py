import os

def rename_models():
    """
    将指定目录中的模型文件重命名为统一格式，例如 0, 1, 2 等（保留原始后缀），并返回模型数量和后缀名。
    """
    directory = "models/parents"  # 模型文件所在的固定目录
    count = 0
    extension = None
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1]
        if not ext:  # 如果文件没有后缀，跳过处理
            print(f"跳过: {old_path}（没有后缀）")
            continue
        if extension is None:
            extension = ext  # 记录第一个文件的后缀
        new_path = os.path.join(directory, f"{count}{ext}")  # 使用 count 作为索引
        os.rename(old_path, new_path)
        print(f"重命名: {old_path} -> {new_path}")
        count += 1
    return count, extension  # 返回数量和后缀
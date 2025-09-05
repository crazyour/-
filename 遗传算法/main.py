import argparse
from src.genetic_optimizer import run_genetic_algorithm
from src.utils import rename_models

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="运行遗传算法")
    parser.add_argument("--generations", type=int, default=100, help="遗传算法的迭代次数")
    parser.add_argument("--parent_population_size", type=int, default=8, help="父代种群的大小")
    parser.add_argument("--children_population_size", type=int, default=30, help="子代种群的大小")
    args = parser.parse_args()

    # 复制进去的模型名字各异，重命名模型成同一的命名格式
    rename_models()

    # 调用遗传算法，传入命令行参数
    run_genetic_algorithm(
        generations=args.generations,
        parent_population_size=args.parent_population_size,
        children_population_size=args.children_population_size
    )
    print("优化完成！")
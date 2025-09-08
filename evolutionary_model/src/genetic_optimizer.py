from src.model_manager import load_model,save_model,get_model_parameters,clear_model,copy_model
import random
import torch
import numpy as np
import os
from evaluate import evaluate

def run_genetic_algorithm(generations, parent_population_size, children_population_size, evaluate, extension):
    """运行遗传算法的主流程"""
    # 初始化种群和适应度列表
    population_pool = {
        'parents': [0] * parent_population_size,
        'children': [0] * children_population_size
    }
    fitness_list = {
        'parents': [0] * parent_population_size,
        'children': [0] * children_population_size
    }
    population_pool_temporary = {'parents_elite': [0] * parent_population_size,
                                 'parents_diverse': [0] * parent_population_size,
                                 'children': [0] * children_population_size}
    fitness_list_temporary = {'parents_elite': [0] * parent_population_size,
                               'parents_diverse': [0] * parent_population_size,
                               'children': [0] * children_population_size}
    # 初始化种群
    for index in range(parent_population_size):
        population_pool['parents'][index] = f"models/parents/{index}{extension}"
    print("初始适应值:", fitness_list['parents'], flush=True)
    print(population_pool)

    # 遗传算法迭代
    for generation in range(generations):
        print(f"第 {generation + 1} 世代:", flush=True)
        population_pool = crossover(children_population_size, parent_population_size, population_pool)
        fitness_list = fitness(population_pool, fitness_list, 'children', evaluate)
        population_pool, fitness_list,population_pool_temporary,fitness_list_temporary = eliminate(population_pool, fitness_list,population_pool_temporary,fitness_list_temporary,parent_population_size)
        print(f"第 {generation + 1} 世代中的最强个体的适应值: {fitness_list['parents']}", flush=True)
    
def crossover(children_population_size, parent_population_size, population_pool):
    all_combinations = [(i, j) for i in range(parent_population_size) for j in range(parent_population_size) if i != j]
    
    # 如果目标数量超过组合数量，反复抽取直到满足目标数量
    if children_population_size > len(all_combinations):
        selected_combinations = []
        while len(selected_combinations) < children_population_size:
            selected_combinations.extend(random.sample(all_combinations, min(children_population_size - len(selected_combinations), len(all_combinations))))
    else:
        selected_combinations = random.sample(all_combinations, children_population_size)

    for index, (model0_idx, model1_idx) in enumerate(selected_combinations): 
        model0_path = population_pool['parents'][model0_idx]
        model1_path = population_pool['parents'][model1_idx]
    
        # 使用 model_loader 加载模型
        model_0 = load_model(model0_path)
        model_1 = load_model(model1_path)

        # 提取参数
        params_0 = get_model_parameters(model_0)
        params_1 = get_model_parameters(model_1)

        print(params_0)
        exit()
        # 交叉操作
        model_2_params = {}
        for name in params_0.keys():
            if name in params_1:
                model_2_params[name] = params_0[name] if random.random() < 0.5 else params_1[name]
            else:
                raise KeyError(f"参数 {name} 在 model_1 中不存在")

        # 设置变异率和变异范围
        mutation_rate = 0.2  # 变异概率
        mutation_range = 0.1  # 变异范围
        for name, param in model_2_params.items():
            if random.random() < mutation_rate:
                if isinstance(param, torch.Tensor) and param.dtype.is_floating_point:
                    noise = (torch.rand_like(param) - 0.5) * 2 * mutation_range
                    param += param * noise
                elif isinstance(param, np.ndarray) and np.issubdtype(param.dtype, np.floating):
                    noise = (np.random.rand(*param.shape) - 0.5) * 2 * mutation_range
                    param += param * noise
                else:
                    print(f"跳过非浮点参数: {name}")

        # 保存包含完整结构和新参数的 model_2
        save_model(model_2_params, f"models/children/{index}.pth")
        population_pool['children'][index] = f"models/children/{index}.pth"
    return population_pool
# 适应值计算
def fitness(population_pool, fitness_list, population_type, evaluate):
    """计算适应值"""
    for i, model_path in enumerate(population_pool[population_type]):
        fitness_list[population_type][i] = evaluate(model_path)
    return fitness_list

# 淘汰函数
def eliminate(population_pool, fitness_list,population_pool_temporary,fitness_list_temporary,parent_population_size):
    combined_scores = [(score, 'parents_elite', i) for i, score in enumerate(fitness_list['parents_elite'])] + \
                      [(score, 'children', i) for i, score in enumerate(fitness_list['children'])]
    sorted_scores = sorted(combined_scores, key=lambda x: x[0], reverse=True)[:parent_population_size]

    for idx, (_, source, i) in enumerate(sorted_scores):
        model_path = population_pool[source][i]
        population_pool_temporary['parents_elite'][idx] = model_path
        fitness_list_temporary['parents_elite'][idx] = fitness_list[source][i]

    return population_pool_temporary, fitness_list_temporary
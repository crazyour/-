from stable_baselines3 import DQN
import gymnasium as gym  # 替换 Gym 为 Gymnasium
import torch

def create_model():
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="human")
    # 初始化 DQN 模型
    model = DQN("MlpPolicy",env)
    #  verbose=1,
    #     learning_rate=0.001,
    #     buffer_size=100000,
    #     batch_size=128,
    #     gamma=0.99,
    #     exploration_fraction=0.4,
    #     exploration_final_eps=0.02,
    #     target_update_interval=250,
    #     train_freq=4,
    #     policy_kwargs=dict(net_arch=[256, 256])
    return model

# 训练模型
def train(num_models):
    """
    训练指定数量的模型，并为每个模型设置有代表性的超参数。
    
    :param num_models: 要训练的模型数量
    """
    for i in range(num_models):
        print(f"正在训练第 {i + 1} 个模型...")
        
        # 动态生成有代表性的超参数
        learning_rate = 0.001 * (1 + i * 0.1)  # 学习率随 i 增加
        gamma = 0.99 - i * 0.01  # 折扣因子随 i 减小
        batch_size = 128 + i * 32  # 批量大小随 i 增加
        buffer_size = 100000 + i * 50000  # 缓冲区大小随 i 增加
        policy_kwargs = dict(net_arch=[256 + i * 64, 256 + i * 64])  # 网络结构随 i 增加
        
        # 创建环境
        env = gym.make("CartPole-v1", render_mode="human")
        
        # 初始化模型
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            policy_kwargs=policy_kwargs
        )
        
        # 训练模型
        model.learn(total_timesteps=80000, log_interval=10)
        
        # 保存模型参数到文件
        param_file = f"models/parents/dqn_cartpole_parameters_{i + 1}.pth"
        torch.save(model.policy.state_dict(), param_file)
        print(f"第 {i + 1} 个模型参数已保存到 {param_file}")

def evaluate(model_params_path, max_episodes=30):
    """
    评估模型性能，计算每回合奖励并输出平均奖励。
    
    :param model_params_path: 保存的模型参数文件路径
    :param max_episodes: 评估回合数
    :return: 平均奖励
    """
    # 使用 create_model 初始化模型
    model = create_model()
    
    # 加载模型参数
    model.policy.load_state_dict(torch.load(model_params_path))
    print(f"模型参数已从 {model_params_path} 加载")
    
    # 获取环境
    env = model.get_env()

    episode_count = 0  # 当前回合计数
    total_rewards = []  # 用于记录每回合的奖励

    while episode_count < max_episodes:
        obs, info = env.reset()
        episode_reward = 0  # 初始化当前回合奖励
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward  # 累积奖励
            if terminated or truncated:
                total_rewards.append(episode_reward)  # 记录当前回合奖励
                print(f"Episode {episode_count + 1}: Reward = {episode_reward}")
                episode_count += 1
                break

    # 输出平均奖励
    average_reward = sum(total_rewards) / max_episodes
    print(f"Average Reward over {max_episodes} episodes: {average_reward}")
    return average_reward

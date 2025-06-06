# Reinforcement Learning: PPO & SAC Implementation

A comprehensive implementation and comparison of **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)** algorithms for various OpenAI Gym environments.

## 🚀 Algorithms Implemented

- **PPO (Proximal Policy Optimization)**: Both continuous and discrete action spaces
- **SAC (Soft Actor-Critic)**: Multiple variants including discrete and continuous implementations

## 🎮 Environments Tested

- **Pendulum-v1** (Continuous Control)
- **CartPole-v1** (Discrete)
- **MountainCar-v0** (Discrete)
- **MountainCarContinuous-v1** (Continuous)
- **Acrobot-v1** (Discrete)

## 📁 Key Files

- `ppo.py` - Main PPO implementation with actor-critic architecture
- `SAC_Continuous.ipynb` - SAC for continuous action spaces
- `SAC_Discrete.ipynb` - SAC for discrete action spaces
- `ppo.ipynb` - PPO experiments and results
- `plots.ipynb` - Performance analysis and visualizations

## 🛠️ Features

- **Modular Architecture**: Separate Actor, Critic, and Memory classes
- **GPU Support**: CUDA acceleration when available
- **Hyperparameter Tuning**: Extensive experimentation documented
- **Performance Comparison**: Direct PPO vs SAC analysis
- **Visualization**: Learning curves and performance plots

## 📊 Results

The repository contains comprehensive experimental results including:
- Learning curves and performance metrics
- Hyperparameter sensitivity analysis
- Algorithm comparison across different environments
- Trained model weights in `results/trained_policies_pth/`

## 🔧 Dependencies

```bash
torch
numpy
matplotlib
gym
```

## 📖 Usage

1. **PPO Training**:
   ```python
   from ppo import PPOAgent
   agent = PPOAgent(make_env, continuous=True, obs_dim=3, act_dim=1, ...)
   agent.train()
   ```

2. **SAC Training**: Open and run the respective Jupyter notebooks
   - `SAC_Continuous.ipynb` for continuous environments
   - `SAC_Discrete.ipynb` for discrete environments

## 📄 Documentation

- `Rapport_RL_2024.pdf` - Comprehensive project report (French)
- `OVERALL HYPERPARAMETERS.pdf` - Hyperparameter documentation
- `Reinforcement_Learning_Poster_Guillaume_Spahr_Mischa_Mez(2024).pdf` - Research poster

## 👥 Authors

**Guillaume Spahr & Mischa Mez**  
EPFL - École Polytechnique Fédérale de Lausanne (2024)

---

*This project demonstrates modern reinforcement learning algorithms with thorough experimental validation and performance analysis.* 
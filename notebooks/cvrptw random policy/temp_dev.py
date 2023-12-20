import torch

from rl4co.envs import CVRPEnv, CVRPTWEnv
from rl4co.models.nn.utils import rollout, random_policy
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer


env = CVRPTWEnv(
    num_loc=50,
    min_loc=0,
    max_loc=150,
    min_demand=1,
    max_demand=10,
    vehicle_capacity=1,
    capacity=10,
    min_time=0,
    max_time=480,
)

# try random policy
batch_size = 1

reward, td, actions = rollout(
    env=env,
    td=env.reset(batch_size=[batch_size]),
    policy=random_policy,
    max_steps=1000,
)
env.get_reward(td, actions)
print("Reward:\n", reward, "\nActions:\n", actions)

# env.render(td, actions)


# # Model: default is AM with REINFORCE and greedy rollout baseline
# print("Start attention model...")
# model = AttentionModel(
#     env, baseline="rollout", train_data_size=100_000, val_data_size=10_000
# )

# # Greedy rollouts over untrained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# td_init = env.reset(batch_size=[3]).to(device)
# model = model.to(device)
# out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

# # Plotting
# print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
# for td, actions in zip(td_init, out["actions"].cpu()):
#     env.render(td, actions)

# # The RL4CO trainer is a wrapper around PyTorch Lightning's `Trainer` class which adds some functionality and more efficient defaults
# trainer = RL4COTrainer(
#     max_epochs=3,
#     accelerator="gpu",
#     devices=1,
#     logger=None,
# )

# # fit model
# trainer.fit(model)

# # Testing

# # Greedy rollouts over trained model (same states as previous plot)
# model = model.to(device)
# out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

# # Plotting
# print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
# for td, actions in zip(td_init, out["actions"].cpu()):
#     env.render(td, actions)
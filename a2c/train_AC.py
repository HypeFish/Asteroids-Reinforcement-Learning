import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from AC_ppo import ActorCriticPPO, device, process_observation

# Environment settings
env_name = "Asteroids-v4"
env = gym.make(env_name, obs_type='grayscale')
state_dim = 84 * 84  # Preprocessed frame dimensions
action_dim = env.action_space.n

# Hyperparameters
lr_actor = 0.0001
lr_critic = 0.001
gamma = 0.99
epochs = 500
batch_size = 64

# Initialize Actor-Critic PPO model
agent = ActorCriticPPO(state_dim, action_dim, lr_actor, lr_critic)
agent.to(device)

# Initialize lists to store losses
actor_losses = []
critic_losses = []

print("Training full agent")

# Training loop
for epoch in tqdm(range(epochs)):
    states = []
    actions = []
    rewards = []
    dones = []
    values = []

    state, _ = env.reset()
    state = process_observation(state)
    while True:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = process_observation(next_state)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated)
        with torch.no_grad():
            values.append(agent.critic(state.unsqueeze(0)).item())

        state = next_state
        if terminated or truncated:
            discounted_rewards = agent.discounted_rewards(rewards, gamma)
            # Assuming values is a list or array of values for each state in the episode
            values = np.array(values)
            # Reshape values to match the shape of discounted_rewards
            values = values.reshape(-1, 1)
            advantages = discounted_rewards - values
            actor_loss, critic_loss = agent.train(states, actions, advantages, discounted_rewards)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            agent.num_games += 1
            agent.scores.append(sum(rewards))
            break

env.close()  

# Plot losses
plt.plot(actor_losses, label='Actor Loss')
plt.plot(critic_losses, label='Critic Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Actor-Critic PPO Losses')
plt.legend()
plt.savefig('losses_no_restrict.png')
plt.cla()

# Plot scores
plt.plot(agent.scores)
plt.xlabel('Game')
plt.ylabel('Score')
plt.title('Game Scores')
plt.savefig('scores_no_restrict.png')
plt.cla()

# Plot average scores
average_scores = torch.mean(torch.tensor(agent.scores).view(-1, 1000), dim=1).numpy()
plt.plot(average_scores)
plt.xlabel('Game')
plt.ylabel('Average Score')
plt.title('Average Game Scores')
plt.savefig('average_scores_no_restrict.png')
plt.cla()


# Save model
torch.save(agent.state_dict(), "agent_no_restrict.pth")
print("Model saved")


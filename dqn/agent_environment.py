"""Episode-based agent-environment interaction loop."""

import numpy as np
import torch


def agent_environment_episode_loop(agent, env, num_episodes, track_q=False):
    """Run ``num_episodes`` training episodes.

    Returns:
        (episode_returns, mean_q_predictions). ``mean_q_predictions[i]`` is the mean
        online-network Q-value of the actions taken in episode i, or ``None`` when
        ``track_q`` is False.
    """
    episode_returns = []
    mean_q_predictions = []

    for _ in range(num_episodes):
        episode_q_values = []
        observation, info = env.reset()
        terminate, truncate = False, False
        episode_return = 0
        while not (terminate or truncate):
            action = agent.act(observation)
            if track_q:
                processed_obs = agent.input_preprocessor(observation)
                obs_tensor = torch.from_numpy(processed_obs).float().unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    all_q_values = agent.q_network(obs_tensor)
                episode_q_values.append(all_q_values[0, action].item())
            next_observation, reward, terminate, truncate, info = env.step(action)
            agent.process_transition(next_observation, reward, terminate, truncate)
            observation = next_observation
            episode_return += reward
        if track_q and len(episode_q_values) > 0:
            mean_q_predictions.append(np.mean(episode_q_values))
        episode_returns.append(episode_return)

    if track_q:
        return episode_returns, mean_q_predictions
    return episode_returns, None

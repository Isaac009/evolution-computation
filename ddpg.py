import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline
from tsp_ga import *

from ddpg_agent import Agent

parameters = None

# exit()
# Environment or settings that will action == state 
# env = gym.make('Humanoid-v3')
# env.seed(10)
# print(env.observation_space)
# print(env.action_space)
agent = Agent(state_size=3, action_size=3, random_seed=10)

def ddpg(n_episodes=20, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        state = np.array([100, 20, 0.01])
        # agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done = step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            parameters = state
            geneticAlgorithm(population=cityList, popSize=parameters[0], eliteSize=parameters[1], mutationRate=parameters[2], generations=500)
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

# state = env.reset()
# agent.reset()   
# while True:
#     action = agent.act(state)
#     env.render()
#     next_state, reward, done, _ = env.step(action)
#     state = next_state
#     if done:
#         break
        
# env.close()
# import gym
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
agent = Agent(state_size=2, action_size=2, random_seed=10)

def ddpg(n_tasks=200, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    score = 0
    state = np.array([20, 0.1])
    population=cityList
    popSize=100
    eliteSize=20
    mutationRate=0.01 
    generations=10
    action = state
    dist_0 = 9999999
    dist = 0

    for i_episode in range(1, n_tasks+1):
        
        # agent.reset()
        
        # def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
        pop = initialPopulation(popSize, population)
        print()
        print("Initial distance: " + str(1 / rankRoutes(pop, 0)[0][1]))
        print()
        rewards = []
        
        for i in range(0, generations):
            pop = nextGeneration(pop, eliteSize=int(action[0]), mutationRate=action[1], gen=i)
            dist = 1 / rankRoutes(pop, i)[0][1]
        rewards.append(dist)
            # print(f"{i} generation length {dist}")

        action = agent.act(state)
        print(action)
        print()

        final_dst = 1 / rankRoutes(pop, i)[0][1]
        diff = dist - dist_0
        dist_0 = dist
        if diff < 0:
            reward = 1
        else:
            reward = -1
        # reward = (1/final_dst) * 100
        
        next_state = action
        done = 0
        score += reward
        print("Reward: ", reward)
        # next_state, reward, done = step(action)
        agent.step(state, action, reward, next_state, done)
        state = action
        print("Different in distance: " + str(diff))
        bestRouteIndex = rankRoutes(pop, 100)[0][0]
        bestRoute = pop[bestRouteIndex]


    #     for t in range(max_t):
            # action = agent.act(state)
    #         next_state, reward, done = step(action)
    #         agent.step(state, action, reward, next_state, done)
    #         state = next_state
    #         parameters = state
    #         pop = initialPopulation(popSize, population)
    #         print("Initial distance: " + str(1 / rankRoutes(pop, 0)[0][1]))
            
    #         for i in range(0, generations):
    #             pop = nextGeneration(pop, eliteSize, mutationRate, i)
    #         final_dst = 1 / rankRoutes(pop, i)[0][1]
    #         print("Final distance: " + str(final_dst))
    #         bestRouteIndex = rankRoutes(pop, 100)[0][0]
    #         bestRoute = pop[bestRouteIndex]

    #         # print("Final distance: ", final_dst)
    #         # return bestRoute, final_dst
    #         # bRoute, final_fit = geneticAlgorithm(population=cityList, popSize=parameters[0], crossoverRate=parameters[1], mutationRate=parameters[2], generations=500)
    #         score += reward
    #         if done:
    #             break
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
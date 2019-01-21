#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import time
import torch

import matplotlib.pyplot as plt

from gym.envs.registration import register

env = gym.make('FrozenLake-v0')

num_episodes = 1000

gamma = 0.95
learning_rate = 0.9

number_of_states = env.observation_space.n 
number_of_actions = env.action_space.n

steps_total = []
rewards_total = []

epsilon = 0.90
epsilon_decay = 0.999
epsilon_final = 0.0001

Q = torch.zeros([number_of_states, number_of_actions])


for i_episode in range(num_episodes):
    
    state = env.reset()
    step = 0
    #for step in range(100):
    episode_reward = 0
    for step in range(100):
    #while True:
        
        step += 1
 	if torch.rand(1)[0] > epsilon: 
		# We will focus on current Q values	
		random_values = Q[state] + torch.rand(1,number_of_actions) / 1000       
        	action = torch.max(random_values,1)[1][0].item()
	else:
		# We do something random
        	action = env.action_space.sample()
       
	if epsilon > epsilon_final: 
		epsilon *= epsilon_decay

        new_state, reward, done, info = env.step(action)
       
	Q[state,action] = (1-learning_rate)*Q[state,action] + learning_rate*(reward + gamma*torch.max(Q[new_state]))


    	episode_reward += reward 

	state = new_state
 
        #time.sleep(0.5)

	if i_episode >= 990:
		env.render()

        #env.render()
	#print reward
        
        #print(new_state)
        #print(info)
        
        if done:
            steps_total.append(step)
	    rewards_total.append(episode_reward)
            print("Episode finished after %i steps" % step )
	    print("Episode ended with %f rewards" % episode_reward)
            break

print(Q)

print('Length: ', len(rewards_total))
 
print('Percentage of successful episodes: {0}'.format(sum(rewards_total)/num_episodes))
print('Percentage of successful episodes (last 100 episodes): {0}'.format(sum(rewards_total[-100:])/100))

        
print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()


plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(0,len(rewards_total)), rewards_total, alpha=0.6, color='green')
plt.show()


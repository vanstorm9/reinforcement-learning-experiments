#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python rl24-PongVideoOutput.py | tee pong-report.txt

# frame input: (240,256,3)
# Previous pong input: (210,160,3)

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time
import os
import numpy as np

import cv2
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import matplotlib.pyplot as plt

plt.style.use('ggplot')

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


directory = './MarioVideos/'
env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: episode_id%20==0)


seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.0001
num_episodes = 5000
startNum = 500
#newModel = False
newModel = True

gamma = 0.99

hidden_layer = 512

replay_mem_size = 100000

# We may want to fix this
batch_size = 32
#batch_size = 96

update_target_frequency = 5000

double_dqn = False

epsilon = 0.5 ##################
egreedy_final = 0.001
#egreedy_decay = 10000

report_interval = 10
score_to_solve = 18

clip_error = True
normalize_image = True

file2save = 'pong_save.pth'
save_model_frequency = 10000
resume_previous_training = False

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

'''
def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon
'''
def calculate_epsilon(steps_done, epsilon):
        if epsilon > egreedy_final:
            epsilon *= 0.9999999
        return epsilon

def load_model():
        return torch.load(file2save)

def save_model(model):
        torch.save(model.state_dict(), file2save)

def preprocess_frame(frame):
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.expand_dims(frame,axis=2)
        frame = frame.transpose((2,0,1))
        frame = np.flip(frame,axis=0).copy()
        frame = torch.from_numpy(frame)
        frame = frame.to(device, dtype=torch.float32)
        frame = frame.unsqueeze(1)

        return frame

def pcot_results():
        plt.figure(figsize=(12,5))
        plt.title("Rewards")
        plt.plot(rewards_total, alpha=0.6, color='red')
        plt.savefig("Mario-results.png")
        plt.close()


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
 
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = ( self.position + 1 ) % self.capacity
        
        
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))
        
        
    def __len__(self):
        return len(self.memory)
        
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
       
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
 
        self.advantage1 = nn.Linear(26*28*64,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)
        
        self.value1 = nn.Linear(26*28*64,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)

        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()
        
        
    def forward(self, x):

        if normalize_image:
                x = x / 255

        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)
       
        #print(output_conv.shape) 
        output_conv = output_conv.view(output_conv.size(0), -1)  # flatten
        #print(output_conv.shape) 
        output_advantage = self.advantage1(output_conv)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)
        
        output_value = self.value1(output_conv)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)
        #print('output value: ',output_value.shape)

        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final
    
class QNet_Agent(object):
    def __init__(self):

        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        if not newModel:
            # load model
            self.nn = torch.load('./model/model.pt',map_location=device)
            self.target_nn = torch.load('./model/model.pt',map_location=device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.number_of_frames = 0
       
        if resume_previous_training and os.path.exists(file2save):
                print("Loading previously saved model . . .")
                self.nn.model.load_state_dict(load_model())


 
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        print('Random for egreedy: ', random_for_egreedy.item(),'>   epsilon: ', epsilon) 
        print('-------------------') 
        if random_for_egreedy.item() > epsilon:      
            print('Greater than epsilon') 
            with torch.no_grad():
                # Convert state to grayscale
                #print('state: ', state.shape)
                #state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                #print('state: ', state.shape)
                #print('state: ', state.shape)
                print('state: ', state.shape)
               
                state = preprocess_frame(state)
                


                print('state: ', state.shape)
                action_from_nn = self.nn(state)
                print('action_from_nn: ',action_from_nn)
                print('action_from_nn.shape: ',action_from_nn.shape)
                action = torch.max(action_from_nn,1)[1]
                print('action: ',action)
                print('action.shape :',action.shape)
                # We need to fix this eventually
                #action = action[0].item()        
                action = action.item()        
        else:
            print('Less than epsilon') 
            #action = env.action_space.sample()
            #frame = frame.transpose((2,0,1))
            print('state.shape: ', state.shape)
            if state.shape[2] == 3:
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                #cv2.imshow('state',state)
                #cv2.waitKey(0)
                state = np.expand_dims(state,axis=2)
            print('state.shape: ', state.shape)
            action = qnet_agent.select_action(state, epsilon)
            print('[',action,']')
            #print(action.shape)

        return action
    
    def optimize(self):
         
        if (len(memory) < batch_size):
            return
        print('-----------------')
        state, action, new_state, reward, done = memory.sample(batch_size)
       
        state = [ preprocess_frame(frame) for frame in state ]
        state = torch.cat(state)

        print('len(new_state):', len(new_state))
         
        new_state = [ preprocess_frame(frame) for frame in new_state ]
        new_state = torch.cat(new_state)
        #print('new_state: ', new_state.shape)

        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        print('new_state.shape:', new_state.shape)
        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            print('new_state_indexes.shape: ',new_state_indexes.shape)
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  
            
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        

        #print('tar: ',max_new_state_values.shape) 
        #print('reward: ',reward.shape)
        #print('test: ', max_new_state_values)

        # We need to fix this
        #target_value = reward + ( 1 - done ) * gamma * max_new_state_values[:32]
        print('max_new_state_values.shape: ',max_new_state_values.shape)
        print('reward.shape: ',reward.shape)
        
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
  

        # We need to fix this
        #predicted_value = self.nn(state[:32]).gather(1, action.unsqueeze(1)).squeeze(1)
        predicted_value = self.nn(state[:32]).gather(1, action.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.number_of_frames % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
       
        if self.number_of_frames % save_model_frequency == 0:
                save_model(self.nn)
 
        self.number_of_frames += 1
        
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])

        
        

memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

rewards_total = []

frames_total = 0 
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(startNum,num_episodes):
    state = env.reset()
   
    # Converting to gray scale

    print('STATE: ',state.shape) 
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = np.expand_dims(state,axis=2)
    print('STATE: ',state.shape) 

    score = 0
    infoStr = 'Starting episode '+ str(i_episode)+ '/ epsilon: '+ str(epsilon)
    print(infoStr,end='')  # Python 3
    #print infoStr, # Python 2
    #for step in range(100):
    while True:
        
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total,epsilon)
        
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)
        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        
        env.render()

        score += reward

        state = new_state
       
        
 
        if done:
            rewards_total.append(score)
            
            mean_reward_100 = sum(rewards_total[-100:])/100
            scoreStr = '/ score:'+str(score)
            print(score, end='\n')   # Python 3
            #print score, # Python 2
            print(score) 

            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
            
            if (i_episode % report_interval == 0 and i_episode > 0):
                
                #plot_results() 
                
                print("\n*** Episode %i *** \
                      \nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                      \nepsilon: %.2f, frames_total: %i" 
                  % 
                  ( i_episode,
                    report_interval,
                    sum(rewards_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(rewards_total)/len(rewards_total),
                    epsilon,
                    frames_total
                          ) 
                  )
                torch.save(qnet_agent.nn,'model.pt')    
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



            break
        

print("\n\n\n\nAverage reward: %.2f" % (sum(rewards_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(rewards_total[-100:])/100))
if solved:
    print("Solved after %i episodes" % solved_after)


env.close()
env.env.close()

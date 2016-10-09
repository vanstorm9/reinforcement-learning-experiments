import sys
import pygame

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

from collections import deque
import numpy as np

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
import random



#c = 'Run'
c = ''


img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

isCrash = False
ifScore = False
mainScore = 0

ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1



def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
   
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    
    return model





def frame_step(input_actions):

	global ifScore
	global isCrash
	global mainScore

	reward = 0.1

	# if statement for hitting ball
	if ifScore:
		reward = 1
		ifScore = False

	# if statement for missing Ball
	if isCrash:
		reward = -1
		isCrash = False1

	terminal = True
	image_data = pygame.surfarray.array3d(pygame.display.get_surface())

	return image_data, reward, terminal

'''
def movePaddle(self, command):
	if command == 'l':
	    self.paddle.left -= 5
	    if self.paddle.left < 0:
	        self.paddle.left = 0

	elif command == 'r':
	    self.paddle.left += 5
	    if self.paddle.left > MAX_PADDLE_X:
	        self.paddle.left = MAX_PADDLE_X
'''
'''
def trainModel(model):

    global c

    # open up a game state to communicate with emulator
    #game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    if c == 'Run':
	OBSERVE = 999999999    #We keep observe, never train
	epsilon = FINAL_EPSILON
	print ("Now we load weight")
	model.load_weights("model.h5")
	adam = Adam(lr=1e-6)
	model.compile(loss='mse',optimizer=adam)
	print ("Weight load successfully")    
    else:                       #We go to training mode
	OBSERVE = OBSERVATION
	epsilon = INITIAL_EPSILON

    t = 0
    while (True):
	loss = 0
	Q_sa = 0
	action_index = 0
	r_t = 0
	a_t = np.zeros([ACTIONS])
	#choose an action epsilon greedy
	if t % FRAME_PER_ACTION == 0:
	    if random.random() <= epsilon:
		print("----------Random Action----------")
		action_index = random.randrange(ACTIONS)
		a_t[action_index] = 1
	    else:
		q = model.predict(s_t)       #input a stack of 4 images, get the prediction
		max_Q = np.argmax(q)
		action_index = max_Q
		a_t[max_Q] = 1

	#We reduced the epsilon gradually
	if epsilon > FINAL_EPSILON and t > OBSERVE:
	    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

	#run the selected action and observed next state and reward
	x_t1_colored, r_t, terminal = frame_step(a_t)

	x_t1 = skimage.color.rgb2gray(x_t1_colored)
	x_t1 = skimage.transform.resize(x_t1,(80,80))
	x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

	x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
	s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

	# store the transition in D
	D.append((s_t, action_index, r_t, s_t1, terminal))
	if len(D) > REPLAY_MEMORY:
	    D.popleft()

	#only train if done observing
	if t > OBSERVE:
	    #sample a minibatch to train on
	    minibatch = random.sample(D, BATCH)

	    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
	    targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

	    #Now we do the experience replay
	    for i in range(0, len(minibatch)):
		state_t = minibatch[i][0]
		action_t = minibatch[i][1]   #This is action index
		reward_t = minibatch[i][2]
		state_t1 = minibatch[i][3]
		terminal = minibatch[i][4]
		# if terminated, only equals reward

		inputs[i:i + 1] = state_t    #I saved down s_t

		targets[i] = model.predict(state_t)  # Hitting each buttom probability
		Q_sa = model.predict(state_t1)

		if terminal:
		    targets[i, action_t] = reward_t
		else:
		    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

	    # targets2 = normalize(targets)
	    loss += model.train_on_batch(inputs, targets)

	s_t = s_t1
	t = t + 1
	
	# save progress every 10000 iterations
	if t % 100 == 0:
	    print("Now we save model")
	    model.save_weights("model.h5", overwrite=True)
	    with open("model.json", "w") as outfile:
		json.dump(model.to_json(), outfile)

	# print info
	state = ""
	if t <= OBSERVE:
	    state = "observe"
	elif t > OBSERVE and t <= OBSERVE + EXPLORE:
	    state = "explore"
	else:
	    state = "train"

	if action_index == 0:
		movePaddle('l')
	elif action_index == 1:
		movePaddle('r')


	print("TIMESTEP", t, "/ STATE", state, \
	    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
	    "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

'''

SCREEN_SIZE   = 640,480

# Object dimensions
BRICK_WIDTH   = 60
BRICK_HEIGHT  = 15
PADDLE_WIDTH  = 60
PADDLE_HEIGHT = 12
BALL_DIAMETER = 16
BALL_RADIUS   = BALL_DIAMETER / 2

MAX_PADDLE_X = SCREEN_SIZE[0] - PADDLE_WIDTH
MAX_BALL_X   = SCREEN_SIZE[0] - BALL_DIAMETER
MAX_BALL_Y   = SCREEN_SIZE[1] - BALL_DIAMETER

# Paddle Y coordinate
PADDLE_Y = SCREEN_SIZE[1] - PADDLE_HEIGHT - 10

# Color constants
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE  = (0,0,255)
BRICK_COLOR = (200,200,0)

# State constants
STATE_BALL_IN_PADDLE = 0
STATE_PLAYING = 1
STATE_WON = 2
STATE_GAME_OVER = 3

class Bricka:

    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("bricka (a breakout clone by codeNtronix.com)")
        
        self.clock = pygame.time.Clock()

        if pygame.font:
            self.font = pygame.font.Font(None,30)
        else:
            self.font = None

        self.init_game()

        
    def init_game(self):
        self.lives = 3
        self.score = 0
        self.state = STATE_BALL_IN_PADDLE

        self.paddle   = pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
        self.ball     = pygame.Rect(300,PADDLE_Y - BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)

        self.ball_vel = [5,-5]

        self.create_bricks()
        

    def create_bricks(self):
        y_ofs = 35
        self.bricks = []
        for i in range(7):
            x_ofs = 35
            for j in range(8):
                self.bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
                x_ofs += BRICK_WIDTH + 10
            y_ofs += BRICK_HEIGHT + 5

    def draw_bricks(self):
        for brick in self.bricks:
            pygame.draw.rect(self.screen, BRICK_COLOR, brick)




    ########### Key controls #############       
    def check_input(self):
        keys = pygame.key.get_pressed()
        




        if keys[pygame.K_LEFT]:
            self.paddle.left -= 5
            if self.paddle.left < 0:
                self.paddle.left = 0

        if keys[pygame.K_RIGHT]:
            self.paddle.left += 5
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X

        if keys[pygame.K_SPACE] and self.state == STATE_BALL_IN_PADDLE:
            self.ball_vel = [5,-5]
            self.state = STATE_PLAYING
        elif keys[pygame.K_RETURN] and (self.state == STATE_GAME_OVER or self.state == STATE_WON):
            self.init_game()



	
	###################################   







    def move_ball(self):
	global mainScore
        self.ball.left += self.ball_vel[0]
        self.ball.top  += self.ball_vel[1]

        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left >= MAX_BALL_X:
            self.ball.left = MAX_BALL_X
            self.ball_vel[0] = -self.ball_vel[0]
        
        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top >= MAX_BALL_Y:            
            self.ball.top = MAX_BALL_Y
            self.ball_vel[1] = -self.ball_vel[1]

    def handle_collisions(self):
        for brick in self.bricks:
            if self.ball.colliderect(brick):
                self.score += 3
		mainScore += 3
		ifScore = True
                self.ball_vel[1] = -self.ball_vel[1]
                self.bricks.remove(brick)
                break

        if len(self.bricks) == 0:
            self.state = STATE_WON
            
        if self.ball.colliderect(self.paddle):
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top > self.paddle.top:
            self.lives -= 1
            if self.lives > 0:
                self.state = STATE_BALL_IN_PADDLE
            else:
                self.state = STATE_GAME_OVER
		isCrash = True

    def show_stats(self):
        if self.font:
            font_surface = self.font.render("SCORE: " + str(self.score) + " LIVES: " + str(self.lives), False, WHITE)
            self.screen.blit(font_surface, (205,5))

    def show_message(self,message):
        if self.font:
            size = self.font.size(message)
            font_surface = self.font.render(message,False, WHITE)
            x = (SCREEN_SIZE[0] - size[0]) / 2
            y = (SCREEN_SIZE[1] - size[1]) / 2
            self.screen.blit(font_surface, (x,y))
        
    def movePaddle(self, command):
	if command == 'l':
	    print 'MOVING LEFT'
	    self.paddle.left -= 5
	    if self.paddle.left < 0:
	        self.paddle.left = 0

	elif command == 'r':
	    print 'MOVING RIGHT'
	    self.paddle.left += 5
	    if self.paddle.left > MAX_PADDLE_X:
	        self.paddle.left = MAX_PADDLE_X

    def trainModel(self,model):



	    global c



	    # open up a game state to communicate with emulator

	    #game_state = game.GameState()



	    # store the previous observations in replay memory

	    D = deque()



	    # get the first state by doing nothing and preprocess the image to 80x80x4

	    do_nothing = np.zeros(ACTIONS)

	    do_nothing[0] = 1

	    x_t, r_0, terminal = frame_step(do_nothing)



	    x_t = skimage.color.rgb2gray(x_t)

	    x_t = skimage.transform.resize(x_t,(80,80))

	    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))



	    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)



	    #In Keras, need to reshape

	    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])



	    if c == 'Run':

		OBSERVE = 999999999    #We keep observe, never train

		epsilon = FINAL_EPSILON

		print ("Now we load weight")

		model.load_weights("model.h5")
		adam = Adam(lr=1e-6)
		model.compile(loss='mse',optimizer=adam)
		print ("Weight load successfully")    
	    else:                       #We go to training mode
		OBSERVE = OBSERVATION
		epsilon = INITIAL_EPSILON

	    t = 0
	    while (True):


	        for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit

                self.clock.tick(50)
                self.screen.fill(BLACK)
                self.check_input()

		
	        ## While game is playing ##
                if self.state == STATE_PLAYING:



                    self.move_ball()
                    self.handle_collisions()






                elif self.state == STATE_BALL_IN_PADDLE:
                    self.ball.left = self.paddle.left + self.paddle.width / 2
                    self.ball.top  = self.paddle.top - self.ball.height
                    self.show_message("PRESS SPACE TO LAUNCH THE BALL")

		    '''		
		    self.ball_vel = [5,-5]
            	    self.state = STATE_PLAYING
		    '''

                elif self.state == STATE_GAME_OVER:
                    self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
                elif self.state == STATE_WON:
                    self.show_message("YOU WON! PRESS ENTER TO PLAY AGAIN")
                



                self.draw_bricks()

                # Draw paddle
                pygame.draw.rect(self.screen, BLUE, self.paddle)

                # Draw ball
                pygame.draw.circle(self.screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)

                self.show_stats()

                pygame.display.flip()













		loss = 0
		Q_sa = 0
		action_index = 0
		r_t = 0
		a_t = np.zeros([ACTIONS])
		#choose an action epsilon greedy
		if t % FRAME_PER_ACTION == 0:
		    if random.random() <= epsilon:
			print("----------Random Action----------")
			action_index = random.randrange(ACTIONS)
			a_t[action_index] = 1
		    else:
			q = model.predict(s_t)       #input a stack of 4 images, get the prediction
			max_Q = np.argmax(q)
			action_index = max_Q
			a_t[max_Q] = 1


		#We reduced the epsilon gradually
		if epsilon > FINAL_EPSILON and t > OBSERVE:
		    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		#run the selected action and observed next state and reward
		x_t1_colored, r_t, terminal = frame_step(a_t)

		x_t1 = skimage.color.rgb2gray(x_t1_colored)
		x_t1 = skimage.transform.resize(x_t1,(80,80))
		x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

		x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
		s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

		# store the transition in D
		D.append((s_t, action_index, r_t, s_t1, terminal))
		if len(D) > REPLAY_MEMORY:
		    D.popleft()

		#only train if done observing

		if t > OBSERVE:
		    #sample a minibatch to train on
		    minibatch = random.sample(D, BATCH)

		    inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
		    targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

		    #Now we do the experience replay
		    for i in range(0, len(minibatch)):
			state_t = minibatch[i][0]
			action_t = minibatch[i][1]   #This is action index
			reward_t = minibatch[i][2]
			state_t1 = minibatch[i][3]
			terminal = minibatch[i][4]
			# if terminated, only equals reward

			inputs[i:i + 1] = state_t    #I saved down s_t

			targets[i] = model.predict(state_t)  # Hitting each buttom probability
			Q_sa = model.predict(state_t1)


			if terminal:
			    targets[i, action_t] = reward_t
			else:
			    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

		    # targets2 = normalize(targets)
		    loss += model.train_on_batch(inputs, targets)

		s_t = s_t1
		t = t + 1
	
		# save progress every 10000 iterations
		if t % 100 == 0:
		    print("Now we save model")
		    model.save_weights("model.h5", overwrite=True)
		    with open("model.json", "w") as outfile:
			json.dump(model.to_json(), outfile)

		# print info
		state = ""
		if t <= OBSERVE:

		    state = "observe"
		elif t > OBSERVE and t <= OBSERVE + EXPLORE:
		    state = "explore"
		else:
		    state = "train"

		print action_index
		if action_index == 0:
			self.movePaddle('l')
		elif action_index == 1:
			self.movePaddle('r')


		print("TIMESTEP", t, "/ STATE", state, \
		    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
		    "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

	    print("Episode finished!")
	    print("************************")







    ############# WHILE LOOP THAT RUNS EVERYTHING #############
    def run(self):
	
	


        while 1:
	    '''            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit

            self.clock.tick(50)
            self.screen.fill(BLACK)
            self.check_input()

		
	    ## While game is playing ##
            if self.state == STATE_PLAYING:



                self.move_ball()
                self.handle_collisions()






            elif self.state == STATE_BALL_IN_PADDLE:
                self.ball.left = self.paddle.left + self.paddle.width / 2
                self.ball.top  = self.paddle.top - self.ball.height
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")

		


            elif self.state == STATE_GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
            elif self.state == STATE_WON:
                self.show_message("YOU WON! PRESS ENTER TO PLAY AGAIN")
                



            self.draw_bricks()

            # Draw paddle
            pygame.draw.rect(self.screen, BLUE, self.paddle)

            # Draw ball
            pygame.draw.circle(self.screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)

            self.show_stats()

            pygame.display.flip()
	    '''
            model = buildmodel()
	
	    self.trainModel(model)
	    
	    #self.movePaddle('l')
		#######################################

if __name__ == "__main__":
    Bricka().run()

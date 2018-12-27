from worldState import *
import random

episodeLimit = 100

alpha =  0.9
gamma = 0.9 
moveProb = 0.8


penalty = -0.05

randMoveProb = (1.0-moveProb)/2.0

def viewGrid(world):
	for row in world:
		print row

def getStateReward(row,col,direction):
	# Check bounderies and obstacles
	
	if direction == 'up':
		posIndex = row-1	
		if posIndex < 0:	
			return grid[row][col]
		pos = grid[posIndex][col]

	elif direction == 'down':
		posIndex = row+1	
		if posIndex >= len(grid):	
			return grid[row][col]
		pos = grid[posIndex][col]
	elif direction == 'left':
		posIndex = col-1	
		if posIndex < 0:	
			return grid[row][col]
		pos = grid[row][posIndex]
	elif direction == 'right':
		posIndex = col+1	
		if posIndex >= len(grid[0]):	
			return grid[row][col]
		pos = grid[row][posIndex]

	# Determines if it is a normal valid state
	if type(pos) is float:
		return pos

	# Checks if terminal state
	if pos[0] == '+':
		return int(pos[1:])
	if pos[0] == '-':
		return int(pos[1:])*-1


	# Probably an obstacle then
	return grid[row][col]

def calculateAction(row,col):
	# We calculate the next action
	
	# Check up
	upValue = alpha*(moveProb*(getStateReward(row,col,'up')) +  randMoveProb*(getStateReward(row,col,'left')) + randMoveProb*(getStateReward(row,col,'right'))    )
	# Check down
	downValue = alpha*(moveProb*(getStateReward(row,col,'down')) +  randMoveProb*(getStateReward(row,col,'left')) + randMoveProb*(getStateReward(row,col,'right'))    )

	# Check left
	leftValue = alpha*(moveProb*(getStateReward(row,col,'left')) +  randMoveProb*(getStateReward(row,col,'up')) + randMoveProb*(getStateReward(row,col,'down'))    )
	# Check right
	rightValue = alpha*(moveProb*(getStateReward(row,col,'right')) +  randMoveProb*(getStateReward(row,col,'up')) + randMoveProb*(getStateReward(row,col,'down'))    )

	evalList = [['up',upValue],['down',downValue],['left',leftValue],['right',rightValue]]

	maxVal = max(evalList, key=lambda x: x[1])
	#print maxVal, ' ', (row,col)

	direction, value = maxVal	

	return direction, value


def nextMove(coord,direction):
	# Need a conditional to determine terminal state
	reward = -0.05

	# each position contains a row of Q-values
	row,col = coord
	pos = grid[row][col]
	currUp, currDown, currLeft, currRight = pos

	# For up
	if row - 1 < 0 or grid[row-1][col] == 'OBS':
		actionUp = grid[row][col]
		coordUp = (row,col)
		currUp = currUp + alpha*(reward + gamma*max(actionUp)  - currUp)		
	elif isinstance(grid[row-1][col], list):
		actionUp = grid[row-1][col]
		coordUp = (row-1,col)
		currUp = currUp + alpha*(reward + gamma*max(actionUp)  - currUp)		
	else:
		# We reached a terminal state	
		# We figure out what terminal state it is
		reward = 0
		coordUp = (row-1,col)
		termPos = grid[row-1][col]
		if termPos[0] == '+':
			reward = int(termPos[1:])
		if termPos[0] == '-':
			reward = int(termPos[1:])*-1
	
		currUp = currUp + alpha*(reward - currUp)		


	# For down
	if row >= len(grid)- 1 or grid[row+1][col] == 'OBS':
		actionDown = grid[row][col]
		coordDown = (row,col)
		currDown = currDown + alpha*(reward + gamma*max(actionDown)  - currDown)		
	elif isinstance(grid[row+1][col], list):
		actionDown = grid[row+1][col]
		coordDown = (row+1,col)
		currDown = currDown + alpha*(reward + gamma*max(actionDown)  - currDown)		
	else:
		# We reached a terminal state	
		# We figure out what terminal state it is
		reward = 0
		coordDown = (row+1,col)
		termPos = grid[row+1][col]
		if termPos[0] == '+':
			reward = int(termPos[1:])
		if termPos[0] == '-':
			reward = int(termPos[1:])*-1
	
		currDown = currDown + alpha*(reward - currDown)		
	# left
	if col - 1 < 0 or grid[row][col-1] == 'OBS':
		actionLeft = grid[row][col]
		coordLeft = (row,col)
		currLeft = currLeft + alpha*(reward + gamma*max(actionLeft)  - currLeft)		
	elif isinstance(grid[row][col-1], list):
		actionLeft = grid[row][col-1]
		coordLeft = (row,col-1)
		currLeft = currLeft + alpha*(reward + gamma*max(actionLeft)  - currLeft)		
	else:
		# We reached a terminal state	
		# We figure out what terminal state it is
		reward = 0
		coordLeft = (row,col-1)
		termPos = grid[row][col-1]
		if termPos[0] == '+':
			reward = int(termPos[1:])
		if termPos[0] == '-':
			reward = int(termPos[1:])*-1
	
		currLeft = currLeft + alpha*(reward - currLeft)		


	# For right
	if col >= len(grid[0]) - 1 or grid[row][col+1] == 'OBS':
		actionRight = grid[row][col]
		coordRight = (row,col)
		currRight = currRight + alpha*(reward + gamma*max(actionRight)  - currRight)		
	elif isinstance(grid[row][col+1], list):
		actionRight = grid[row][col+1]
		coordRight = (row,col+1)
		currRight = currRight + alpha*(reward + gamma*max(actionRight)  - currRight)		
	else:
		# We reached a terminal state	
		# We figure out what terminal state it is
		reward = 0
		coordRight = (row,col+1)
		termPos = grid[row][col+1]
		if termPos[0] == '+':
			reward = int(termPos[1:])
		if termPos[0] == '-':
			reward = int(termPos[1:])*-1
	
		currRight = currRight + alpha*(reward - currRight)		


	evalList = [['up',currUp, coordUp, 0], ['down',currDown, coordDown, 1], ['left',currLeft, coordLeft,2], ['right',currRight, coordRight,3]]


	if direction != 'max':
		if direction == 'up':
			return evalList[0]
		elif direction == 'down':
			return evalList[1]
		elif direction == 'left':
			return evalList[2]
		elif direction == 'right':
			return evalList[3]



	# find max	
	return max(evalList, key=lambda x: x[1])


def runTraining():

	episodeNum = 0

	while True:
		# Iterate through every episode
		print '----------- ', episodeNum, ' --------------'

		movementList = []
		startx,starty = startState

		coord = (startx,starty)
		pos = grid[startx][starty]

		terminalState = False
		while not terminalState:
		
			if random.randint(0,10) > 9:
				randDirAr = ['up','down','left','right']
				randIndex = random.randint(0,3)
				direction, value, coord, index = nextMove(coord,randDirAr[randIndex])
			else:
				direction, value, coord, index = nextMove(coord,'max')
			pos[index] = round(value,2)
			pos = grid[coord[0]][coord[1]]
			movementList.append([direction, coord])

			#print movementList
			#print '\n\n'
			#viewGrid(grid)
			#print '\n\n'
			if pos[0] == '+' or pos[0] == '-':
				# terminal state
				terminalState = True
				break


		episodeNum += 1
		print movementList

		if episodeNum > episodeLimit:
			print '\n\n\n\n'
			print 'After ', episodeNum - 1, ' iterations: '
			return

	return


print 'Inital State'
viewGrid(grid)
print '\n\n\n'

runTraining()

print '\n\n\n'
viewGrid(grid)
print '\n\n\n'




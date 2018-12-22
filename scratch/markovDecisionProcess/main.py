from worldState import *

iterNum = 100

alpha =  0.40 
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
'''
def updateStateValue(row,col,direction,value):

	return
'''
def policyEvaluation():

	# We iterate through everything
	for rowNum,row in enumerate(grid):
		for colNum,pos in enumerate(row):
			# If it is not a valid state, we ignore
			if type(pos) is not float:
				continue
			# We have to check all four directions
			direction, value = calculateAction(rowNum,colNum)

			# Update states
			grid[rowNum][colNum] = round(penalty + alpha*value,2) 			
			gridDir[rowNum][colNum] = direction
'''	
def move(direction):
	if direction == 'up':
		
'''

def attemptStage():
	startX, startY = startState
	pos = grid[startX][startY]
	intentDir = gridDir[startX][startY]

	
	pos = move(intentDir)


print 'Inital State'
viewGrid(grid)
print '\n\n\n'

for i in range(0,iterNum):
	policyEvaluation()

print 'After ', iterNum, ' iterations: '
viewGrid(grid)
print '\n'
viewGrid(gridDir)




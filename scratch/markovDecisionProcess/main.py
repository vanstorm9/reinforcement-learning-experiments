grid = [[0.0,0.0,0.0,'+1'],
	[0.0,'OBS',0.0,'-1'],
	[0.0,0.0,0.0,0.0]]

startState = (2,2)

alpha =  1.0 
moveProb = 0.8
randMoveProb = (1.0-moveProb)/2.0

def viewGrid():
	for row in grid:
		print row

def getStateReward(row,col,direction):
	# Check bounderies and obstacles
	
	if direction == 'up':
		posIndex = row-1	
		pos = grid[posIndex][col]

		if posIndex < 0:	
			return grid[row][col]
	elif direction == 'down':
		posIndex = row+1	
		pos = grid[posIndex][col]

		if posIndex < len(grid):	
			return grid[row][col]
	elif direction == 'left':
		posIndex = col-1	
		pos = grid[row][posIndex]

		if posIndex < 0:	
			return grid[row][col]
	elif direction == 'left':
		posIndex = col+1	
		pos = grid[row][posIndex]

		if posIndex < len(grid[0]):	
			return grid[row][col]
	
	

	# Checks if terminal state
	if pos[0] == '+' or pos[0]=='-':
		return int(pos[1])
	if pos[0] == '-':
		return int(pos[1])*-1

	# Determines if it is a normal valid state
	if type(pos) is float:
		return pos

	# Probably an obstacle then
	return grid[row][col]

def calculateAction(row,col):
	# We calculate the next action
	
	# Check up
	upValue = alpha*(moveProb*(getStateReward(row,col,'up')) +  randMoveProb(getStateReward(row,col,'left')) + randMoveProb(getStateReward(row,col,'right'))    )

	# Check down
	downValue = alpha*(moveProb*(getStateReward(row,col,'down')) +  randMoveProb(getStateReward(row,col,'left')) + randMoveProb(getStateReward(row,col,'right'))    )





def policyEvaluation():

	# We iterate through everything
	for rowNum,row in enumerate(grid):
		for pos in enumerate(row):
			# If it is not a valid state, we ignore
			if type(pos) is not float:
				continue
			# We have to check all four directions
			

	


#viewGrid()
policyEvaluation()




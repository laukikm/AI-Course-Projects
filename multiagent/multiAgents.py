# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from game import Directions

from game import Actions
import util

#import ipdb

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def bfs(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #The following lines are mine
    
    from game import Directions
    #import ipdb
    
    #Problem is an object of class SearchProblem
    start_state=problem.getStartState()
    state=(start_state,-1,0) #-1 and -100 is to indicate that this is the start state
    
    fringe_queue=util.Queue()
    #fringe_queue.push([state],0)
    explored_states=util.Stack()
    
    successor_states=problem.getSuccessors(state[0])
    #successor_state_for_this_state[state]=successor_states

    
    for i in successor_states:
        temp=[state]
        temp.append(i)
        fringe_queue.push(temp)
        #costs[i[0]]=i[2]
    #ipdb.set_trace()    
        
    traj=fringe_queue.pop()
    curr_state=traj[-1]
    nodes_expanded=[state[0]]
    
    while(not problem.isGoalState(curr_state[0])):
        
        if(curr_state[0] not in nodes_expanded):
            nodes_expanded.append(curr_state[0])
            successor_states=problem.getSuccessors(curr_state[0])
              
            for next_state in successor_states:
                temp=traj[:]
                temp.append(next_state)
                cost=0
                if next_state[0] not in nodes_expanded:
                    
                    fringe_queue.push(temp)
        #if curr_state[0]==[5,1,1,1,1,1]:
        #    ipdb.set_trace()

        #ipdb.set_trace()   
        #print curr_state     
        traj=fringe_queue.pop()
        curr_state=traj[-1]
            
    #Loop has exited=> curr_state=goalState
    action_sequence=[]
    for i in traj[1:]:
        action_sequence.append(i[1])
    
    return action_sequence

    util.raiseNotDefined()


class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost





class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        import ipdb
        #ipdb.set_trace()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        #print newPos
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Get the distance of the closest food particle
        def monster_direction(currentGameState):
            pacman_pos=currentGameState.getPacmanPosition()
            ghost_states=currentGameState.getGhostStates()
            
            forbidden_directions=[]
            for ghost in ghost_states:
                ghost_pos=ghost.getPosition()
                rel_pos=(ghost_pos[0]-pacman_pos[0],ghost_pos[1]-pacman_pos[1])
                #relative_ghost_positions.append(rel_pos)
                if(rel_pos[0]>0):
                    forbidden_directions.append('East')
                elif(rel_pos[0]<0):
                    forbidden_directions.append('West')
                if(rel_pos[1]>0):
                    forbidden_directions.append('North')
                elif(rel_pos[1]<0):
                    forbidden_directions.append('South')
            return forbidden_directions


        #import ipdb
        min_distance=float('inf')
        for i in newFood.asList():
            if(min_distance>manhattanDistance(newPos,i)):
                min_distance=manhattanDistance(newPos,i)
        scared_times_sum=sum(newScaredTimes)
        min_ghost_distance=float('inf')

        #ipdb.set_trace()
        for j in range(len(newGhostStates)):
            i=newGhostStates[j]
            scared_time=newScaredTimes[j]
            if(min_ghost_distance>manhattanDistance(newPos,i.getPosition()) and scared_time==0):
                min_ghost_distance=manhattanDistance(newPos,i.getPosition())

        if(len(newFood.asList())==0):
            return float('inf')

        if(min_ghost_distance<4):
            #ipdb.set_trace()
            forbidden_directions=monster_direction(currentGameState)
            #print -20000/(min_ghost_distance+0.01)*(action in forbidden_directions)-2*min_distance-100*len(newFood.asList())
            if(action in forbidden_directions or action=='Stay'):
                return -200000/(min_ghost_distance+0.01)-2*min_distance-100*len(newFood.asList())
            else:
                return -2*min_distance-100*len(newFood.asList())

        return -10/min_ghost_distance-2*min_distance-100*len(newFood.asList())




def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    #def stupid_fn(self,state):
        #return state.getScore()

    def get_minimax_utility(self,current_state,agent_to_act,current_depth):
        n_agents=current_state.getNumAgents()
        #import ipdb
        #Define base case
        if(agent_to_act==0 and current_depth==self.depth): #If all agents have acted, it is the turn of the pacman
            return self.evaluationFunction(current_state)
        else:
            legal_actions=current_state.getLegalActions(agent_to_act)
            successor_states=[current_state.generateSuccessor(agent_to_act,action) for action in legal_actions]
            #print len(successor_states)
            if(agent_to_act==n_agents-1): #Means depth is definitely less than self.depth
                #ipdb.set_trace()
                if(len(legal_actions)>0):
                    utility=min([self.get_minimax_utility(state,0,current_depth+1) for state in successor_states])
                else:#If there is no legal action the agent can take, simply move to the next agent
                    utility=self.get_minimax_utility(current_state,0,current_depth+1)
            elif(agent_to_act==0):
                if(len(legal_actions)>0):    
                    utility=max([self.get_minimax_utility(state,agent_to_act+1,current_depth) for state in successor_states])
                else:
                    utility=self.get_minimax_utility(current_state,agent_to_act+1,current_depth)
            else:
                if(len(legal_actions)>0):
                    utility=min([self.get_minimax_utility(state,agent_to_act+1,current_depth) for state in successor_states])
                else:
                    utility=self.get_minimax_utility(current_state,agent_to_act+1,current_depth)   
        return utility


        
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #The following lines are mine
        import ipdb
        n_agents=gameState.getNumAgents()
        legal_actions=gameState.getLegalActions(0) #Get the legal actions of the pacman
        successor_states=[gameState.generateSuccessor(0,action) for action in legal_actions] #All these are at a depth 1

        utilities=[self.get_minimax_utility(current_state,1,0) for current_state in successor_states] #This is a recursive function
        
        max_index=utilities.index(max(utilities))
        return legal_actions[max_index]
        #ipdb.set_trace()

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def get_next_agent(self,n_agents,agent_to_act,depth):
        next_agent=agent_to_act+1
        if(agent_to_act==n_agents-1):
            next_agent=0
            depth+=1
        return next_agent,depth

    
    def getAction(self,gameState):
        def max_value(current_state,agent_to_act,current_depth,alpha,beta):
            if(current_state.isWin() or current_state.isLose() or (agent_to_act==0 and current_depth==self.depth)):
                return current_state.getScore() #Return score on the terminal state

            legal_actions=current_state.getLegalActions(agent_to_act)
            n_agents=current_state.getNumAgents()
            next_agent,next_depth=self.get_next_agent(n_agents,agent_to_act,current_depth)

            v=-float('inf')
            best_action=[]
            best_score=v
            for action in legal_actions:
                v=max(v,min_value(current_state.generateSuccessor(agent_to_act,action),next_agent,next_depth,alpha,beta))
                if(v>best_score):
                    best_action=action
                    best_score=v

                alpha=max(alpha,v)
                if(v>beta):
                    return v
            
            if agent_to_act==0 and current_depth==0: #If agent==0 and depth==0, beta=inf, so this function will not end prematurely
                return best_action
            else:
                return v

        def min_value(current_state,agent_to_act,current_depth,alpha,beta):
            

            if(current_state.isWin() or current_state.isLose() or (agent_to_act==0 and current_depth==self.depth)):
                return current_state.getScore() #Return score on the terminal state

            legal_actions=current_state.getLegalActions(agent_to_act)
            n_agents=current_state.getNumAgents()
            next_agent,next_depth=self.get_next_agent(n_agents,agent_to_act,current_depth)

            v=float('inf')

            for action in legal_actions:
                import ipdb
                #ipdb.set_trace()
                if(next_agent==0):

                    v=min(v,max_value(current_state.generateSuccessor(agent_to_act,action),next_agent,next_depth,alpha,beta))
                else:
                    v=min(v,min_value(current_state.generateSuccessor(agent_to_act,action),next_agent,next_depth,alpha,beta))
                beta=min(beta,v)
                if(v<alpha):
                    return v
            
            return v
        alpha,beta=-float('inf'),float('inf')
        return max_value(gameState,0,0,alpha,beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def get_expectimax_utility(self,current_state,agent_to_act,current_depth):
        n_agents=current_state.getNumAgents()
        #import ipdb
        #Define base case
        if(agent_to_act==0 and current_depth==self.depth): #If all agents have acted, it is the turn of the pacman
            return self.evaluationFunction(current_state)
        else:
            legal_actions=current_state.getLegalActions(agent_to_act)
            successor_states=[current_state.generateSuccessor(agent_to_act,action) for action in legal_actions]
            #print len(successor_states)
            if(agent_to_act==n_agents-1): #Means depth is definitely less than self.depth
                #ipdb.set_trace() 
                if(len(legal_actions)>0):
                    utilities=[self.get_expectimax_utility(state,0,current_depth+1) for state in successor_states]
                    utility=sum(utilities)/len(utilities) #It is impossible for len(utilities) to be zero
                else:#If there is no legal action the agent can take, simply move to the next agent
                    utility=self.get_expectimax_utility(current_state,0,current_depth+1)
            elif(agent_to_act==0):
                if(len(legal_actions)>0):    
                    utility=max([self.get_expectimax_utility(state,agent_to_act+1,current_depth) for state in successor_states])
                else:
                    utility=self.get_expectimax_utility(current_state,agent_to_act+1,current_depth)
            else:
                if(len(legal_actions)>0):
                    utilities=[self.get_expectimax_utility(state,agent_to_act+1,current_depth) for state in successor_states]
                    utility=sum(utilities)/len(utilities)
                else:
                    utility=self.get_expectimax_utility(current_state,agent_to_act+1,current_depth)   
        return utility

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #The following lines are mine
        legal_actions=gameState.getLegalActions(0) #Get the legal actions of the pacman
        successor_states=[gameState.generateSuccessor(0,action) for action in legal_actions] #All these are at a depth 1

        utilities=[self.get_expectimax_utility(current_state,1,0) for current_state in successor_states] #This is a recursive function
        max_index=utilities.index(max(utilities))
        return legal_actions[max_index]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    

    def mazeDistance(point1, point2, gameState):
            """
            Returns the maze distance between any two points, using the search functions
            you have already built. The gameState can be any game state -- Pacman's
            position in that state is ignored.

            Example usage: mazeDistance( (2,4), (5,6), gameState)

            This might be a useful helper function for your ApproximateSearchAgent.
            """
            x1, y1 = point1
            x2, y2 = point2
            walls = gameState.getWalls()
            assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
            assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
            prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
            return len(bfs(prob))
    


    pacman_pos=currentGameState.getPacmanPosition()
    food=currentGameState.getFood()

    

    ghosts=currentGameState.getGhostStates()

    min_distance=float('inf')


    nearest_point=(1,1)
    for i in food.asList():
        if(min_distance>manhattanDistance(pacman_pos,i)):
            min_distance=manhattanDistance(pacman_pos,i)
            nearest_point=i
    #scared_times_sum=sum(newScaredTimes)
    min_ghost_distance=1000

    
    for j in range(len(ghosts)):
        i=ghosts[j]
        scared_time=i.scaredTimer
        if(min_ghost_distance>manhattanDistance(pacman_pos,i.getPosition()) and scared_time==0):
            min_ghost_distance=manhattanDistance(pacman_pos,i.getPosition())

    

    min_distance=mazeDistance(pacman_pos,nearest_point,currentGameState)
    
    if(len(food.asList())==0):
        #import ipdb
        #ipdb.set_trace()
        return 20*min_ghost_distance #float('inf')

    #if(len(food.asList())==1 and min_distance==1):
        #ipdb.set_trace()

    if(min_ghost_distance<2):
        return -1000/(0.01+min_ghost_distance)-2*(0.01+min_distance)-500*len(food.asList())
    
    #print 20.0*(1+min_distance)-500.0*len(food.asList()),-10.0/(1+min_ghost_distance)
    return -10.0/min_ghost_distance-2.0*(0.01+min_distance)-500.0*len(food.asList())
    
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


# Goal of Project

The main goal of the project is to test three search algorithms (BFS, IDS, and A*) on a given environment and compare their performances.

# Part 1. Environment and Test Cases

There is an agent in a maze in form of a grid of cells. each cell can be open or blocked. empty cells can contain an arbitary number of
balls (probably zero). the agent has a limited capacity knapsack. in each move agent can move to an open neighbour cell, pick a ball from a cell or drop a ball in current cell.
dropping balls doesnt count in it moves. the goal is to each ball to a certain cell declared in test case description

test cases describe environment in bellow format:

&emsp;- $N$ & $M$ : number of rows and columns of maze  

&emsp;- $S_i$ & $S_j$ : start position

&emsp;- $F_i$ & $F_j$ : finish position

&emsp;- $C$ : knapsack capacity

&emsp;- $K$ : number of balls

&emsp;- $K$ lines containing two pairs of numbers describing initial position and desired position for each ball

&emsp;- $N$ row of __<font color=red>"-"</font>__ for open cells and __<font color=red>"*"</font>__ for blocked cells describing maze's map

### 1.0 Imports


```python
from copy import deepcopy
import numpy as np
import pandas as pd
from fnvhash import fnv1a_32
import random, time, itertools, hashlib
```

### 1.1 Loading Test Cases


```python
def load_test_case(path):
    fd = open(path,"r")
    test_case = [line.strip() for line in fd.read().splitlines()]
    return test_case
```


```python
TESTS_PATH = [
    "./tests/1.txt",
    "./tests/2.txt",
    "./tests/3.txt",
    "./tests/4.txt",
]

test_cases = [load_test_case(path) for path in TESTS_PATH]

for i,t in enumerate(test_cases):
    print('\n\n\nTest Case {}:'.format(i+1))
    for row in t:
        print(*row, sep='')
```

    
    
    
    Test Case 1:
    5 7
    1 0
    2 6
    1
    3
    1 2 2 5
    2 3 3 3
    3 2 1 1
    * * * * * * *
    - - - - - - *
    * * * - - - -
    * - - - - - *
    * * * * * * *
    
    
    
    Test Case 2:
    50 50
    24 24
    48 48
    1
    0
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    
    
    Test Case 3:
    4 42
    1 1
    2 40
    1
    10
    1 30 2 30
    1 31 2 31
    1 32 2 32
    1 33 2 33
    1 34 2 34
    1 35 2 35
    1 36 2 36
    1 37 2 37
    1 38 2 38
    1 39 2 39
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    
    
    Test Case 4:
    30 30
    1 0
    28 29
    1
    0
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    - - - - * - - - - - - - - - - - - - - - - - - - - - - - - *
    * - * - * - * * * * * * - * - * * * - * * * * - * * - * - *
    * - * - - - - - - - - - - * - * - * - * - - * - * - - * - *
    * - * - * * * * - - * * * * - * - - - * - - * * * * - * - *
    * - * - - - - - - - - - - - - * - * - * - - - - - - - * - *
    * - * * * * * * * * - * * * - * - * - * - * * * * * - - - *
    * - - - - - - - - * - * - * - - - * - - - - - - - * - * - *
    * * * * - * * * - * - * - * - * - * - * * * * * - * - * - *
    * - - - - * - - - * - * - * - * - * - - - - - - - * - * - *
    * - - - - - - - - * - * - * - - - * - * - - - * * * - - - *
    * * - * * * * * * * - * - * - * - - - * - - - - - - - * - *
    * - - * - - * - - - - * - * - * - - - * - - * * * * * * - *
    * - - - - - * - * * * * - * - * * - * * * - * - - - * - - *
    * - * * * * * - * - - - - * - - - - - - - - * - * - * - * *
    * - - * - - - - * - * * - * * * * * - * * * * - * - * - - *
    * * - * - * * - * - * - - * - - - - - * - - - - * - * * - *
    * - - * - * - - - - * * * * - - * * * * - * * * * - * - - *
    * - * * - * - - - - - - - - - - * - - - - * - - * - * - * *
    * - - * - * * * * * * * * - - - * - * * * * - - * - * - - *
    * * - * - - - - - - - - * - - - * - * - - - - - * - * * - *
    * - - * * * * * * * * - * * * * * - * - - * * * * - * - - *
    * - - * - - - * - - * - - - - - - - * - - * - - - - * - * *
    * - * * - * - * - - * * * * * * * * * - - * - * * * * - - *
    * - - - - * - - - - - - * - - - - - - - - * - * - - * - - *
    * * * * * * * - * * * - * - * - * * * * - * - * - * * * - *
    * - - - - - * - * - * - * - - - * - - * - * - * - - - - - *
    * - * * * * * - * - * - * * * - * * - * - * - * * * * * * *
    * - - - - - - - - - * - - - - - - - - - - * - - - - - - - -
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# Part 2. Modeling


## 2.1 Agent:
in all algorithms the agent is equivalent to a state and represents a node in search graph
## &emsp;- __Attributes__:
&emsp;        __Position__: position of agent in environment

&emsp;        __knapsack__: a set of balls that agent is carrying right now

&emsp;        __delivered_balls__: a set of balls that agent is has delivered to their destination

&emsp;        __moves_history__: number of agent moves from start. it was a set of move records initialy but it was slow on copyinig Agent objects in large test cases.

## &emsp;- __Methods__:

&emsp;        __state_hash()__: a unique identifier string representing the state of agent based on current position and moves history

&emsp;        __move(newposition)__: move agent to a neighbour cell with $newposition$ coordinates

&emsp;        __pick_ball(ball_index)__: pick ball with number $ball$ $index$ and add it to knapsack (if its possible)

&emsp;        __drop_ball(ball_index)__: drop ball with number $ball$ $index$ in current cell.

&emsp; and some getter functions ... 


```python
class Agent():
    def __init__(self, test_case):
        self.__position = [int(c) for  c in test_case[1].split(' ')]
        self.__knapsack_capacity = int(test_case[3].split(' ')[0])
        self.__knapsack = set()
        self.__delivered_balls = set()
        self.__moves_history = 0
        
    def __lt__(self, other):
        return True
        
    def state_hash(self):
        #state = '{:03d}'.format(self.__position[0])+"00"+'{:03d}'.format(self.__position[1])+"00"+("".join([str(b) for b in self.__knapsack]))+"00"+("".join([str(b) for b in self.__delivered_balls]))
        state = str(self.__position) + "-" + str(self.__knapsack) + "-" + str(self.__delivered_balls)
        return state
        #return int(hashlib.sha1(state.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
        
    def drop_ball(self, ball):
        if (ball in self.__knapsack) and (ball not in self.__delivered_balls):
            self.__knapsack.discard(ball)
            self.__delivered_balls.add(ball)
    
    def pick_ball(self, ball):
        if (len(self.__knapsack) < self.__knapsack_capacity) and (ball not in self.__knapsack) and (ball not in self.__delivered_balls):
            self.__knapsack.add(ball)
            self.__moves_history += 1
            
    def move(self, new_position):
        self.__position = new_position
        self.__moves_history += 1
            
    def position(self):
        return self.__position
            
    def delivered_balls(self):
        return self.__delivered_balls 
    
    def knapsack(self):
        return self.__knapsack
    
    def report(self):
        return self.__moves_history
    
    def knapsack_capacity(self):
        return self.__knapsack_capacity
```

## 2.2 Environment:
## &emsp;- __Attributes__:
&emsp;        __rows__ & __columns__: dimensions of map

&emsp;        __agent_destination__

&emsp;        __number_of_balls__

&emsp;        __balls_source__: a list containing initial coordination of balls
    
&emsp;        __balls_destination__: a list containing final coordination of balls
    
&emsp;        __balls_source_of[ $i$ ][ $j$ ]__: a set of ball with initial coordination $(i,j)$

&emsp;        __balls_destination_of[ $i$ ][ $j$ ]__: a set of ball with final coordination $(i,j)$

&emsp;        __map__: map of maze containing "-" and "*" characters

## &emsp;- __Methods__:

&emsp;        __is_finished(agent)__: check if an agent is in final state, i.e, it has delivered all balls and is in Agent's destination (specified in test case) in map

&emsp;        __get_valid_moves(agent)__: returns a list of (method, args) pairs containing all valid moves the agent can take in this step. agent has two type of moves: moving to a valid neighbour cell that is showed by ("Agent.move", neighbour_coordination) or picking a ball that is showed by ("Agent.pick_ball", ball_index)

&emsp;        __check_balls_to_deliver(agent)__: checks if there is somoe balls in agents's knapsack with destination of current cell. if there is some the environment forces agent to drop them.

&emsp; and some getter functions ... 


```python
class Env():
    
    AGENT_MOVES = [ [0,1], [1,0], [0,-1], [-1,0] ]
    
    def __init__(self, test_case) :
        self.__rows, self.__columns = map(int,test_case[0].split(' '))
        self.__agent_destination = [int(c) for  c in test_case[2].split(' ')]
        self.__number_of_balls = int(test_case[4].split(' ')[0])
        self.__balls_source = []
        self.__balls_destination = []
        self.__ball_source_of= [[set() for j in range(self.__columns)] for i in range(self.__rows)]
        self.__ball_destination_of= [[set() for j in range(self.__columns)] for i in range(self.__rows)]
        for i in range(self.__number_of_balls):
            start_row, start_column, finish_row, finish_column = [int(x) for x in test_case[5+i].split(' ')]
            self.__balls_source.append( (start_row, start_column) )
            self.__balls_destination.append( (finish_row, finish_column) )
            self.__ball_source_of[start_row][start_column].add(i)
            self.__ball_destination_of[finish_row][finish_column].add(i)
        for i in range (self.__rows):
            for j in range(self.__columns):
                self.__ball_source_of[i][j] = self.__ball_source_of[i][j]-self.__ball_destination_of[i][j]
                
        self.__map = [line.split() for line in test_case[5+self.__number_of_balls:]]
        
    def is_finished(self, agent):
        if len(agent.delivered_balls()) == self.__number_of_balls and agent.position() == self.__agent_destination :
            return True
        else:
            return False
    
    def get_valid_moves(self, agent):
        valid_moves = []
        agent_row = agent.position()[0]
        agent_column = agent.position()[1]
        for move in self.AGENT_MOVES:
            if (agent_row+move[0] in range(0,self.__rows)) and (agent_column+move[1] in range(0,self.__columns)) and \
                                                   (self.__map[agent_row+move[0]][agent_column+move[1]] == '-') :
                valid_moves.append({"method":Agent.move , "args": [agent_row+move[0], agent_column+move[1]]})
        valid_moves += [{"method":Agent.pick_ball, "args": ball} for ball in (self.__ball_source_of[agent_row][agent_column] - agent.delivered_balls() - agent.knapsack())]
        random.shuffle(valid_moves)
        return valid_moves
        
    
    def check_balls_to_deliver(self, agent):
        agent_row = agent.position()[0]
        agent_column = agent.position()[1]
        delivered_balls = self.__ball_destination_of[agent_row][agent_column].intersection(agent.knapsack())
        for ball in delivered_balls:
            agent.drop_ball(ball)
            
    def number_of_balls(self):
        return self.__number_of_balls
    
    def ball_source(self, ball):
        return self.__balls_source[ball]
    
    def ball_destination(self, ball):
        return self.__balls_destination[ball]
    
    def untouched_balls(self, agent):
        balls = set(range(self.__number_of_balls))
        return balls - agent.knapsack() - agent.delivered_balls()
        
    def agent_destination(self):
        return self.__agent_destination
        
```

# Part 3. Search Algorithms

### 3.0- Some Helper Functions


```python
def print_moves(moves_history, node_id, env):
    if moves_history[node_id][0]==-1:
        return set()
    knapsack = print_moves(moves_history, moves_history[node_id][0], env)
    if moves_history[node_id][1]["method"] == Agent.move:
        print("move to {}".format(moves_history[node_id][1]["args"]))
        new_knapsack = set()
        for ball in knapsack:
            if list(env.ball_destination(ball)) == moves_history[node_id][1]["args"] :
                print("drop ball {}".format(ball))
                continue
            new_knapsack.add(ball)
        knapsack=new_knapsack
    else :
        print("pick ball {}".format(moves_history[node_id][1]["args"] ))
        knapsack.add(moves_history[node_id][1]["args"])
    return knapsack
        
def manhattan_ditance(point1, point2):
    return abs(point1[0]-point2[0])+abs(point1[1]-point2[1])
```

## 3.1 BFS


```python
def BFS(env, agent):
    queue = []
    visited = set()
    moves_history = [(-1,None)]
    queue.append( (agent,0) )
    visited.add(agent.state_hash())
    produced_states = 0
    expanded_states = 0
    nodes = 0
    while( queue ):
        node = queue.pop(0)
        agent = node[0]
        node_id = node[1]
        expanded_states += 1
        env.check_balls_to_deliver(agent)
        if env.is_finished(agent):
            return {"answer_depth":agent.report(), "produced_states":produced_states, "expanded_states":expanded_states, "last_node":node_id, "moves_history": moves_history}
        for move in env.get_valid_moves(agent):  
            new_agent = deepcopy(agent)
            method=move["method"]
            args=move["args"]
            method(new_agent,args)
            produced_states += 1
            if new_agent.state_hash() not in visited :
                nodes += 1
                queue.append( (new_agent,nodes) )
                visited.add(new_agent.state_hash())
                moves_history.append( (node_id, move) )
```

## 3.2 IDS
 __DLS__ function implements DFS search with a specific depth limit
 
 __IDS__ iterates over depth limit to find optimal depth answer. the original IDS algorithm starts from depth zero and increase it by 1 each step and finds the lowest(optimal) answer. but this implementation uses binary search to find optimal depth. starting from 1 each time DLS fails to find answer with current limit, depth limit increases exponentialy to find an upper bound for answer.
after that IDS uses binary search to find optimal depth
 that is in range [$UpperBound$/2, $UpperBound$]


```python
def DLS(env, agent, visited, depth, depth_limit, moves_history, last_visit_depth):
    visited.add(agent.state_hash())
    result = {"answer_depth": None, "produced_states":0, "expanded_states":1, "moves_history": moves_history.copy()}
    if depth > depth_limit:
        return result
    env.check_balls_to_deliver(agent)
    if env.is_finished(agent):
        result["answer_depth"] = agent.report()
        return result
    for move in env.get_valid_moves(agent) :
        new_agent = deepcopy(agent)
        method=move["method"]
        args=move["args"]
        method(new_agent,args)
        result["produced_states"] += 1
        if (new_agent.state_hash() not in visited) or ((new_agent.state_hash() in visited) and (last_visit_depth[new_agent.state_hash()] > depth+1)):
            moves_history.append( (len(moves_history)-1, move) )
            last_visit_depth[new_agent.state_hash()]=depth+1
            child_result = DLS(env, new_agent, visited, depth+1, depth_limit, moves_history, last_visit_depth)
            result["produced_states"] += child_result["produced_states"]
            result["expanded_states"] += child_result["expanded_states"]
            del(moves_history[-1])
            if child_result["answer_depth"] is not None:
                result["answer_depth"] = child_result["answer_depth"]
                result["moves_history"] = child_result["moves_history"]
                return child_result
    return result
```


```python
def IDS(env, agent):
    depth_limit_lower_bound = 0
    depth_limit_upper_bound = -1
    depth_limit_test = 1
    result = {"answer_depth":agent.report(), "produced_states":0, "expanded_states":0}
    while(True):
        moves_history = [(-1,None)]
#         print("depth limit:",depth_limit_test)
        visited = set()
        last_visit_depth = {}
        last_visit_depth[agent.state_hash()] = 0
        phase_result = DLS(env, deepcopy(agent), visited, 0, depth_limit_test, moves_history, last_visit_depth)
        result["produced_states"] += phase_result["produced_states"]
        result["expanded_states"] += phase_result["expanded_states"]
        if phase_result["answer_depth"] is not None:
            if depth_limit_test==depth_limit_lower_bound+1 :
#                 print("Found Optimal Answer!")
                result["answer_depth"]=phase_result["answer_depth"]
                result["moves_history"]=phase_result["moves_history"]
                result["last_node"]=len(phase_result["moves_history"])-1
                return result
            else:
#                 print("Found An Answer :)")
                depth_limit_upper_bound=depth_limit_test+1
        else:
#             print("Answer Not Found :(")
            depth_limit_lower_bound=depth_limit_test
        if depth_limit_upper_bound==-1:
            depth_limit_test*=2
        else:
            depth_limit_test = (depth_limit_lower_bound+depth_limit_upper_bound)//2
```

## 3.3 A*

this function implements A* algorithm with given heuristic function. since it implements graph search if heuristic function is __admissible__ (doesnt estimate cost more than reality) the search isnt optimal (but tree search would be optimal) and if heuristic function is __consistent__ the graph search would be optimal


```python
def A_star(env, agent, heuristic):
    queue = set()
    visited = set()
    last_visit_depth = {}
    moves_history = [(-1,None)]
    queue.add( (heuristic(env, agent), 0, agent, 0) ) # g(n)+h(n), depth, agent, node_id
    visited.add( agent.state_hash() )
    last_visit_depth[agent.state_hash()]=0
    produced_states = 0
    expanded_states = 0
    nodes = 0
    while( queue ):
        node = min(queue)
        queue.discard(node)
        depth = node[1]
        agent = node[2]
        node_id = node[3]
        expanded_states += 1
        env.check_balls_to_deliver(agent)
        if env.is_finished(agent):
            return {"answer_depth":agent.report(), "produced_states":produced_states, "expanded_states":expanded_states, "last_node": node_id, "moves_history": moves_history}
        for move in env.get_valid_moves(agent):  
            new_agent = deepcopy(agent)
            method=move["method"]
            args=move["args"]
            method(new_agent,args)
            produced_states += 1
            if (new_agent.state_hash() not in visited) or ((new_agent.state_hash() in visited) and (last_visit_depth[new_agent.state_hash()] > depth+1)):
                nodes += 1
                queue.add( (depth+1+agent.A_star_alpha*heuristic(env, new_agent), depth+1, new_agent, nodes) )
                visited.add(new_agent.state_hash())
                moves_history.append( (node_id, move) )
                last_visit_depth[new_agent.state_hash()]=depth+1
                

```

### 3.3.1 First Heuristic For A*

This heuristic is a lower bound for cost function. It calculates best case in which by picking farthest ball (or delivering it) all other jobs get done thus it admissible.


```python
def heuristic1(env, agent):
    h=0
    for ball in env.untouched_balls(agent):
        h = max(h, manhattan_ditance(agent.position(), env.ball_source(ball)))
    for ball in agent.knapsack():
         h = max(h, manhattan_ditance(env.ball_source(ball), env.ball_destination(ball)))
    h = max(h, manhattan_ditance(agent.position(), env.agent_destination()))
    return h
```

### 3.3.2 Second Heuristic For A*

This heuristic calculates three costs
    1 - delivering balls starting from closest destination to agent's current position and go on by order of manhattan distance of balls destinations.
    2 - picking up balls starting from closest ball source to agent's current position and go on by order of manhattan distance of balls sources.
    3 - goinf toward agents final destination
    
in each state any of these three cost functions has a weight in final heuristic value of state.
    as number of balls in agents knapsack goes higher the weight of delivering balls (__deliver_weight__) increases to motivate agent to search for destination states. and when number of balls in knapsack is low weight of picking balls (__picking_weight__)increases to motivate agent for picking balls.
    on the other hand as number of delivered balls increases the weight of agent itself increases (__agent_destination_weight__) and agents final destination gets higher weight in calculations.
    
consistency of heuristic is difficult to prove because formula is intuitive but it can be shown that moving greedily from inner circle to outer circles from current position gives lowest total cost in manhattan distances. based on this observation we used this strategy to estimate moves needed to collect all balls and also deliver balls in knapsack.
weightings guaranty that the estimated cost cant get higher than real path cost. also in each move by adding one unit to g(n) if we deliver more than one ball it can make heuristic inconsistent but we should notice that in this situation the __deliver_weight__ and __agent_tasks_weight__ increases and balance the estimated value.


```python
def heuristic2(env, agent):
    
    balls = list(env.untouched_balls(agent))
    balls.sort(key=lambda ball:manhattan_ditance(agent.position(), env.ball_source(ball)))
    if len(balls) == 0:
        h1=0
    else:
        h1 = manhattan_ditance(agent.position(), env.ball_source(balls[0]))
        for i in range(1,len(balls)):
            h1 += manhattan_ditance( env.ball_source(balls[i-1]), env.ball_source(balls[i]) )

    balls = list(agent.knapsack())
    balls.sort(key=lambda ball:manhattan_ditance(agent.position(), env.ball_destination(ball)))
    if len(balls) == 0:
        h2=0
    else:
        h2 = manhattan_ditance(agent.position(), env.ball_destination(balls[0]))
        for i in range(1,len(balls)):
            pathCost += manhattan_ditance( env.ball_destination(balls[i-1]), env.ball_destination(balls[i]) )
    if agent.knapsack_capacity() > 0:
        deliver_weight = (len(agent.knapsack())/agent.knapsack_capacity())
    else:
        deliver_weight=0
    pickup_weight = 1-deliver_weight
    if env.number_of_balls() > 0:
        agent_tasks_weight = 1 - len(agent.delivered_balls())/env.number_of_balls()
    else:
        agent_tasks_weight=0
    agent_destination_weight = 1-agent_tasks_weight
    return agent_tasks_weight*(pickup_weight*h1 + deliver_weight*h2) + agent_destination_weight*manhattan_ditance(agent.position(), env.agent_destination())
```

### 3.3.3 A* decorators


```python
def A_start_heuristic_1(env, agent):
    agent.A_star_alpha = 1
    return A_star(env, agent, heuristic1)

def A_start_heuristic_2(env, agent):
    agent.A_star_alpha = 1
    return A_star(env, agent, heuristic2)

def weighted_A_star_1(env, agent):
    agent.A_star_alpha = 1.3
    return A_star(env, agent, heuristic2)

def weighted_A_star_2(env, agent):
    agent.A_star_alpha = 1.6
    return A_star(env, agent, heuristic2)

```

# 4 Comparing Performance of Algortihms


```python
ALGORIITHMS = {"BFS": BFS, "IDS": IDS, "A* with admissible heuristic": A_start_heuristic_1, 
               "A* with consistent heuristic": A_start_heuristic_2, "Weighted A* 1 (alpha=1.3)":weighted_A_star_1,
              "Weighted A* 2 (alpha=1.6)":weighted_A_star_2}

def run(test_case, print_agent_moves=False, runs=3):
    dataframe = pd.DataFrame(columns = ['Algorithm', 'Answer Depth', 'Expanded States', 'Produced States', 'Average Runtime'])
    env = Env(test_case)
    agent = Agent(test_case)
    
    results = []
    
    for search_name, search_function in ALGORIITHMS.items():
        runtimes = []
        expanded_states = []
        produced_states = []
        result = None

        for run in range(runs): 

            start=time.time()
            result = search_function(env, agent)
            finish=time.time()
            expanded_states.append(result["expanded_states"])
            produced_states.append(result["produced_states"])
            runtimes.append(finish-start)
        
        dataframe = dataframe.append({'Algorithm' : search_name,
                                      'Answer Depth': result["answer_depth"],
                                      'Expanded States': int(np.mean(expanded_states)),
                                      'Produced States': int(np.mean(produced_states)),
                                      'Average Runtime': np.mean(runtimes) }, ignore_index=True)
        if print_agent_moves==True:
            print("------ {} answer path -----".format(search_name))
            print_moves(result["moves_history"], result["last_node"], env)
    
    dataframe.set_index('Algorithm')
    
    return dataframe
```


```python
run(test_cases[0], print_agent_moves=True)
```

    ------ BFS answer path -----
    move to [1, 1]
    move to [1, 2]
    move to [1, 3]
    move to [2, 3]
    pick ball 1
    move to [3, 3]
    drop ball 1
    move to [3, 2]
    pick ball 2
    move to [3, 3]
    move to [2, 3]
    move to [1, 3]
    move to [1, 2]
    move to [1, 1]
    drop ball 2
    move to [1, 2]
    pick ball 0
    move to [1, 3]
    move to [2, 3]
    move to [2, 4]
    move to [2, 5]
    drop ball 0
    move to [2, 6]
    ------ IDS answer path -----
    move to [1, 1]
    move to [1, 2]
    move to [1, 3]
    move to [2, 3]
    pick ball 1
    move to [3, 3]
    drop ball 1
    move to [3, 2]
    pick ball 2
    move to [3, 3]
    move to [2, 3]
    move to [1, 3]
    move to [1, 2]
    move to [1, 1]
    drop ball 2
    move to [1, 2]
    pick ball 0
    move to [1, 3]
    move to [1, 4]
    move to [2, 4]
    move to [2, 5]
    drop ball 0
    move to [2, 6]
    ------ A* with admissible heuristic answer path -----
    move to [1, 1]
    move to [1, 2]
    move to [1, 3]
    move to [2, 3]
    pick ball 1
    move to [3, 3]
    drop ball 1
    move to [3, 2]
    pick ball 2
    move to [3, 3]
    move to [2, 3]
    move to [1, 3]
    move to [1, 2]
    move to [1, 1]
    drop ball 2
    move to [1, 2]
    pick ball 0
    move to [1, 3]
    move to [1, 4]
    move to [1, 5]
    move to [2, 5]
    drop ball 0
    move to [2, 6]
    ------ A* with consistent heuristic answer path -----
    move to [1, 1]
    move to [1, 2]
    move to [1, 3]
    move to [2, 3]
    pick ball 1
    move to [3, 3]
    drop ball 1
    move to [3, 2]
    pick ball 2
    move to [3, 3]
    move to [2, 3]
    move to [1, 3]
    move to [1, 2]
    move to [1, 1]
    drop ball 2
    move to [1, 2]
    pick ball 0
    move to [1, 3]
    move to [1, 4]
    move to [2, 4]
    move to [2, 5]
    drop ball 0
    move to [2, 6]
    ------ Weighted A* 1 (alpha=1.3) answer path -----
    move to [1, 1]
    move to [1, 2]
    move to [1, 3]
    move to [2, 3]
    pick ball 1
    move to [3, 3]
    drop ball 1
    move to [3, 2]
    pick ball 2
    move to [3, 3]
    move to [2, 3]
    move to [1, 3]
    move to [1, 2]
    move to [1, 1]
    drop ball 2
    move to [1, 2]
    pick ball 0
    move to [1, 3]
    move to [1, 4]
    move to [2, 4]
    move to [2, 5]
    drop ball 0
    move to [2, 6]
    ------ Weighted A* 2 (alpha=1.6) answer path -----
    move to [1, 1]
    move to [1, 2]
    move to [1, 3]
    move to [2, 3]
    pick ball 1
    move to [3, 3]
    drop ball 1
    move to [3, 2]
    pick ball 2
    move to [3, 3]
    move to [2, 3]
    move to [1, 3]
    move to [1, 2]
    move to [1, 1]
    drop ball 2
    move to [1, 2]
    pick ball 0
    move to [1, 3]
    move to [1, 4]
    move to [2, 4]
    move to [2, 5]
    drop ball 0
    move to [2, 6]





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>Answer Depth</th>
      <th>Expanded States</th>
      <th>Produced States</th>
      <th>Average Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BFS</td>
      <td>20</td>
      <td>253</td>
      <td>644</td>
      <td>0.021725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IDS</td>
      <td>20</td>
      <td>2063</td>
      <td>4632</td>
      <td>0.239389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A* with admissible heuristic</td>
      <td>20</td>
      <td>233</td>
      <td>610</td>
      <td>0.019825</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A* with consistent heuristic</td>
      <td>20</td>
      <td>205</td>
      <td>545</td>
      <td>0.018969</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Weighted A* 1 (alpha=1.3)</td>
      <td>20</td>
      <td>199</td>
      <td>535</td>
      <td>0.018309</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Weighted A* 2 (alpha=1.6)</td>
      <td>20</td>
      <td>207</td>
      <td>557</td>
      <td>0.018740</td>
    </tr>
  </tbody>
</table>
</div>




```python
run(test_cases[1], print_agent_moves=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>Answer Depth</th>
      <th>Expanded States</th>
      <th>Produced States</th>
      <th>Average Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BFS</td>
      <td>48</td>
      <td>2304</td>
      <td>9022</td>
      <td>0.200528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IDS</td>
      <td>48</td>
      <td>117759</td>
      <td>443306</td>
      <td>17.649619</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A* with admissible heuristic</td>
      <td>48</td>
      <td>625</td>
      <td>2448</td>
      <td>0.063068</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A* with consistent heuristic</td>
      <td>48</td>
      <td>625</td>
      <td>2448</td>
      <td>0.065425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Weighted A* 1 (alpha=1.3)</td>
      <td>48</td>
      <td>49</td>
      <td>186</td>
      <td>0.005225</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Weighted A* 2 (alpha=1.6)</td>
      <td>48</td>
      <td>49</td>
      <td>185</td>
      <td>0.005200</td>
    </tr>
  </tbody>
</table>
</div>




```python
run(test_cases[2], print_agent_moves=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>Answer Depth</th>
      <th>Expanded States</th>
      <th>Produced States</th>
      <th>Average Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BFS</td>
      <td>68</td>
      <td>235957</td>
      <td>727914</td>
      <td>21.964026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IDS</td>
      <td>68</td>
      <td>2971517</td>
      <td>8201129</td>
      <td>679.212956</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A* with admissible heuristic</td>
      <td>68</td>
      <td>93456</td>
      <td>295655</td>
      <td>88.233335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A* with consistent heuristic</td>
      <td>68</td>
      <td>69303</td>
      <td>214401</td>
      <td>34.708055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Weighted A* 1 (alpha=1.3)</td>
      <td>68</td>
      <td>27105</td>
      <td>82972</td>
      <td>7.450951</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Weighted A* 2 (alpha=1.6)</td>
      <td>68</td>
      <td>14386</td>
      <td>43722</td>
      <td>2.905774</td>
    </tr>
  </tbody>
</table>
</div>




```python
run(test_cases[3], print_agent_moves=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>Answer Depth</th>
      <th>Expanded States</th>
      <th>Produced States</th>
      <th>Average Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BFS</td>
      <td>92</td>
      <td>457</td>
      <td>1019</td>
      <td>0.031551</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IDS</td>
      <td>92</td>
      <td>14948</td>
      <td>33470</td>
      <td>1.417700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A* with admissible heuristic</td>
      <td>92</td>
      <td>457</td>
      <td>1019</td>
      <td>0.027327</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A* with consistent heuristic</td>
      <td>92</td>
      <td>457</td>
      <td>1019</td>
      <td>0.028784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Weighted A* 1 (alpha=1.3)</td>
      <td>92</td>
      <td>530</td>
      <td>1188</td>
      <td>0.033865</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Weighted A* 2 (alpha=1.6)</td>
      <td>92</td>
      <td>608</td>
      <td>1370</td>
      <td>0.038707</td>
    </tr>
  </tbody>
</table>
</div>



# 5 Discussoin on Results

The main features of the used search algorithmes are compared in the table bellow:

| Algorithm Name | Completeness | Optimality | Time Complexity | Space Complexity |
| --- | --- | --- | --- | --- |
| BFS | Complete | Optimal | O(b^d) | O(b^d) |
| IDS | Complete | Optimal (with step cost 1) | (d+1)b^0 + d b^1 + (d-1)b^2 + … + bd | O(b*d) |
| A* | Complete | Optimal (with consistent heuristic in graph search and admissible in tree search) | Exponential | Exponential |

BFS and IDS algorithms work as expected in this problem. Generally admissible algorithms dont guaranty optimall in graph search and we used graph search in this problem (by maintaining explored set and visited nodes) but admissible heuristic led to optimall answer although heuristic didnt help too much in pruning useless branches and optimizing search time. But consistent heuristic had an better effect on search speed. Different heuristic weighting in A* are helpful in specific problems as we see in second and third test case combining this weighting with a proper heuristic can speed up search significantly.

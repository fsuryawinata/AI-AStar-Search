# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion

from .utils import render_board
from math import sqrt
import heapq


class Node:
    """
    Node class for A* search
    """

    def __init__(self, parent=None, state=None, direction=None, g=0, h=0):
        self.parent = parent
        self.state = state
        self.direction = direction

        self.g = g  # cost so far
        self.h = h  # cost to goal
        self.f = self.g + self.h  # total cost to goal

    def __lt__(self, other):
        return self.f < other.f


def heuristic(curr_state, goal_states):
    """
    Takes current node location and returns the min Euclidian
    distance between current node and goal nodes
    """
    x1, y1 = curr_state
    cost_to_goal = []
    for x2, y2 in goal_states:
        dx = x1 - x2
        dy = y1 - y2
        cost_to_goal.append(sqrt(dx ** 2 + dy ** 2))
    return min(cost_to_goal)


def generateSuccessors(parent, goal_state):
    x, y = parent
    successors = {}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]
    for dx, dy in directions:
        successor = (x + dx, y + dy)

        # Wrap around the hexagon graph if the successor state is outside the boundaries
        if x < 0:
            successors[(6, y)] = (dx, dy)
        elif x > 6:
            successors[(0, y)] = (dx, dy)
        if y < 0:
            successors[(x, 6)] = (dx, dy)
        elif y > 6:
            successors[(x, 0)] = (dx, dy)

        if successor[0] < 0:
            successors[(6, successor[1])] = (dx, dy)
        elif successor[0] > 6:
            successors[(0, successor[1])] = (dx, dy)
        if successor[1] < 0:
            successors[(successor[0], 6)] = (dx, dy)
        elif successor[1] > 6:
            successors[(successor[0], 0)] = (dx, dy)
    return successors


def makeValidSuccessors(x, y):
    newx = x
    newy = y
    if isValid(x, y):
        return x, y
    else:
        if x > 6:
            newx = 0
        elif x < 0:
            newx = 0
        if y > 6:
            newy = 0
        elif y < 0:
            newy = 6
        return newx, newy


def isValid(x, y):
    """
    Check if node is valid
    """
    return (x >= 0) & (x < 7) & (y >= 0) & (y < 7)


def search(input: dict[tuple, tuple]) -> list[tuple]:
    """
    This is the entry point for your submission. The input is a dictionary
    of board cell states, where the keys are tuples of (r, q) coordinates, and
    the values are tuples of (p, k) cell states. The output should be a list of 
    actions, where each action is a tuple of (r, q, dr, dq) coordinates.

    See the specification document for more details.
    """
    # initialise goal states
    goal_states = {}
    for key, value in input.items():
        if value[0] == 'b':
            goal_states[key] = value

    # Find most optimal start node (a red node closest to blue node)
    # print(input)

    # initialise start node
    frontier = []
    explored = set()
    red_node = list(input.keys())[0]
    direction = (0, 0)
    init_node = Node(None, red_node, direction, heuristic(red_node, goal_states))
    heapq.heappush(frontier, init_node)

    path = []
    while frontier:
        curr_node = heapq.heappop(frontier)
        curr_state = curr_node.state
        print(f"Curr state: {curr_state}")

        if curr_state in goal_states:
            # path = []
            while curr_node:
                path.append((curr_state, direction))
                curr_node = curr_node.parent
                if curr_node:
                    curr_state = curr_node.state
                else:
                    None

    explored.add(curr_state)

    successors = generateSuccessors(curr_state, goal_states)
    step_cost = 1
    for successor_state, direction in successors.items():
        if successor_state in explored:
            continue
        print(successor_state)
        successor_cost = curr_node.g + step_cost
        successor_h = heuristic(successor_state, goal_states)
        successor_node = Node(successor_state, curr_node, direction, successor_cost, successor_h)
        heapq.heappush(frontier, successor_node)

    # print(path)

    # The render_board function is useful for debugging -- it will print out a
    # board state in a human-readable format. Try changing the ansi argument 
    # to True to see a colour-coded version (if your terminal supports it).
    print(render_board(input, ansi=False))

    # Here we're returning "hardcoded" actions for the given test.csv file.
    # Of course, you'll need to replace this with an actual solution...
    return [
        (5, 6, -1, 1),
        (3, 1, 0, 1),
        (3, 2, -1, 1),
        (1, 4, 0, -1),
        (1, 3, 0, -1)
    ]

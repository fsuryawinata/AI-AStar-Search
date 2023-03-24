# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion

from .utils import render_board
from math import sqrt
import heapq


class Node:
    """
    Node class for A* search
    """

    def __init__(self, parent=None, state=None, direction=None, power=None, g=0, h=0):
        self.parent = parent
        self.state = state  # location of the node
        self.direction = direction  # direction it took to get here
        self.power = power

        self.g = g  # cost so far
        self.h = h  # cost to goal
        self.f = self.g + self.h  # total cost to goal

    def __lt__(self, other):
        return self.f < other.f


def heuristic(curr_state, goal_states):
    """
    Takes current node location and returns the minimum Euclidian
    distance between current node and goal nodes
    """
    x1, y1 = curr_state
    cost_to_goal = []
    for x2, y2 in goal_states:
        dx = x1 - x2
        dy = y1 - y2
        cost_to_goal.append(sqrt(dx ** 2 + dy ** 2))
    return min(cost_to_goal)


def generateSuccessors(parent_node):
    """
    Generate 6 successors of the parent node
    """
    x, y = parent_node.state
    successors = {}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]
    for dx, dy in directions:
        successor = (x + dx, y + dy)

        # Wrap around the hexagon graph if the successor state is outside the boundaries
        if successor[0] < 0:
            successor = (6, successor[1])
        elif successor[0] > 6:
            successor = (0, successor[1])
        if successor[1] < 0:
            successor = (successor[0], 6)
        elif successor[1] > 6:
            successor = (successor[0], 0)

        # add to dictionary with direction taken
        successors[successor] = (dx, dy)
    return successors


def search(input: dict[tuple, tuple]) -> list[tuple]:
    """
    This is the entry point for your submission. The input is a dictionary
    of board cell states, where the keys are tuples of (r, q) coordinates, and
    the values are tuples of (p, k) cell states. The output should be a list of 
    actions, where each action is a tuple of (r, q, dr, dq) coordinates.

    See the specification document for more details.
    """
    print(render_board(input, ansi=True))
    # initialise goal states
    goal_states = {}
    for key, value in input.items():
        if value[0] == 'b':
            goal_states[key] = value

    # Find most optimal start node (a red node closest to blue node)

    frontier = []
    path = []
    explored = set()

    # Initialise start node
    direction = (0, 0)
    red_node = list(input.keys())[0]
    red_power = list(input.values())[0][1]
    init_node = Node(None, red_node, direction, red_power, heuristic(red_node, goal_states))
    heapq.heappush(frontier, init_node)

    # Run while all goal nodes found
    curr_power = 1
    while goal_states:
        # Run while heap queue exists
        while frontier:
            # Remove current node from queue
            curr_node = heapq.heappop(frontier)
            curr_state = curr_node.state
            curr_power = curr_node.power
            print(curr_state)

            # If goal is found, add to path
            if curr_state in goal_states:
                curr_power += goal_states[curr_state][1]
                goal_states.pop(curr_state)
                print(f"Goal {curr_state} FOUND")
                while curr_node:
                    path.append((curr_state, direction))
                    curr_node = curr_node.parent
                    if curr_node:
                        curr_state = curr_node.state
                    else:
                        None
                break

            # Add to visited nodes
            explored.add(curr_state)

            # Generate successors for current node
            step_cost = 1
            successors = generateSuccessors(curr_node)

            i = 0
            print(f"POWER {curr_power}")
            while i <= curr_power:
                for successor_state, direction in successors.items():
                    if successor_state in explored:
                        continue

                    # Create and add generated nodes into queue
                    successor_cost = curr_node.g + step_cost
                    successor_h = heuristic(successor_state, goal_states)
                    successor_node = Node(curr_node, successor_state, direction, 1,
                                          successor_cost, successor_h)
                    heapq.heappush(frontier, successor_node)
                i += 1

    # Reverses path taken from goal to initial node
    output = []
    for state, direction in path:
        output.append((state[0], state[1], direction[0], direction[1]))
    output.reverse()
    return output

    # The render_board function is useful for debugging -- it will print out a
    # board state in a human-readable format. Try changing the ansi argument
    # to True to see a colour-coded version (if your terminal supports it).

    """
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
    """

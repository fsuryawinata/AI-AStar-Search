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
        # Return greater power if distance to goal is equal
        if self.f == other.f:
            return self.power > other.power
        else:
            return self.f < other.f


def EuclidianDistance(curr_state, goal_states):
    """
    Takes current node location and returns the Euclidian
   distance between current node and goal nodes
    """
    x1, y1 = curr_state
    cost_to_goal = []

    for x2, y2 in goal_states:
        dx = x1 - x2
        dy = y1 - y2
        cost_to_goal.append(sqrt(dx ** 2 + dy ** 2))

    return cost_to_goal


def heuristic(curr_state, goal_states):
    """
    Returns the minimum Euclidian distance between current node and goal nodes
    """
    cost_to_goal = EuclidianDistance(curr_state, goal_states)
    return min(cost_to_goal)


def is_between_goals(curr_state, goal_states):
    """
    Checks if successor is between goals for exclusion
    """
    cost_to_goal = EuclidianDistance(curr_state, goal_states)
    cost_to_goal.sort()
    if len(cost_to_goal) > 1:
        return cost_to_goal[0] == cost_to_goal[1]
    else:
        return False


def generateSuccessors(parent_node):
    """
    Generate the successors of the parent node
    """
    power = parent_node.power
    x, y = parent_node.state
    successors = {}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]

    for dx, dy in directions:
        successor = (x + dx * power, y + dy * power)
        grid_size = 7

        # Wrap around the hexagon graph if the successor state is outside the boundaries
        if successor[0] < 0:
            successor = (abs(successor[0]) % grid_size, successor[1])
        elif successor[0] > 6:
            successor = (successor[0] % grid_size, successor[1])
        if successor[1] < 0:
            successor = (successor[0], abs(successor[0]) % grid_size)
        elif successor[1] > 6:
            successor = (successor[0], successor[1] % grid_size)

        # add to dictionary with direction taken
        successors[successor] = (dx, dy)
    return successors


def checkSuccessor(parent_node, successor_state, goal_states):
    """
    Check if there are goal states between successor and parent state in the a direction
    """
    x_p, y_p = parent_node.state
    x_s, y_s = successor_state
    x_diff = int(abs(x_p - x_s))
    y_diff = int(abs(y_p - y_s))

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]
    if (x_diff != 0) and (y_diff != 0):
        return None
    elif x_diff == 0:
        new_dir = (0, y_diff / abs(y_diff))
        distance_diff = y_diff
    else:
        new_dir = (x_diff / abs(x_diff), 0)
        distance_diff = x_diff

    if new_dir in directions:
        x_dir, y_dir = new_dir

        i = 0
        while i < distance_diff:
            x_s += x_dir
            y_s += y_dir
            if successor_state in goal_states:
                return successor_state, new_dir
            i += 1

    return None


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
    final_path = []
    final_directions = []
    explored = set()

    # Initialise start node
    direction = (0, 0)
    red_node = list(input.keys())[0]
    red_power = list(input.values())[0][1]
    init_node = Node(None, red_node, direction, red_power, heuristic(red_node, goal_states))
    heapq.heappush(frontier, init_node)

    new_start_node = Node(None)
    # Run while all goal nodes found
    while goal_states:
        path = []
        prev_direction = []

        # Reset list with last goal state as new start
        if new_start_node.parent is not None:
            frontier = []
            heapq.heappush(frontier, new_start_node)

        # Run while heap queue exists
        while frontier:
            # Remove current node from queue
            curr_node = heapq.heappop(frontier)
            curr_state = curr_node.state

            # If goal is found, add to path
            if curr_state in goal_states:
                curr_power = goal_states[curr_state][1] + 1
                goal_states.pop(curr_state)
                new_start_node = curr_node
                new_start_node.power = curr_power

                while curr_node:
                    if curr_state not in path:
                        path.append(curr_state)
                        prev_direction.append(curr_node.direction)

                        curr_node = curr_node.parent
                        if curr_node:
                            curr_state = curr_node.state
                        else:
                            None

                # Append to final path and direction for output
                i = 0
                while i < len(prev_direction):
                    if prev_direction[i] == (0, 0):
                        prev_direction.pop(i)
                    i += 1

                path.reverse()
                prev_direction.reverse()
                # print(path)
                #print(prev_direction)

                for steps in path:
                    if steps not in final_path:
                        final_path.append(steps)

                for dir in prev_direction:
                    final_directions.append(dir)
                break

            # Add to visited nodes
            explored.add(curr_state)

            # Generate successors for current node
            step_cost = 1
            successors = generateSuccessors(curr_node)

            for successor_state, direction in successors.items():
                print(goal_states)
                if successor_state in explored:
                    continue
                if is_between_goals(successor_state, goal_states):
                    continue

                # Check if there are any goals in between nodes and add goal found to path
                #print(f"parent state {curr_state}")
                #print(f"before dir of {successor_state}: {direction}")
                if curr_node.power != 1:
                    # print(f"{curr_state} check {successor_state}")
                    goal_found = checkSuccessor(curr_node, successor_state, goal_states)
                    # print(f"goal found {goal_found}")

                    # If goal found, pop
                    if goal_found:
                        #print(f"before {goal_states}")
                        goal_states.pop(goal_found[0])
                        #print(f"After {goal_states}")
                        direction = goal_found[1]

                #print(f"after dir of {successor_state}: {direction}")
                if goal_states:
                    # Create and add generated nodes into queue
                    successor_cost = curr_node.g + step_cost * curr_node.power
                    successor_h = heuristic(successor_state, goal_states)
                    successor_node = Node(curr_node, successor_state, direction, 1,
                                      successor_cost, successor_h)
                    heapq.heappush(frontier, successor_node)

    # Reverses path taken from goal to initial node
    output = []
    i = 0
    while i < len(final_path):
        x, y = final_path[i]
        dir_x, dir_y = final_directions[i]
        output.append((x, y, dir_x, dir_y))
        i += 1

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

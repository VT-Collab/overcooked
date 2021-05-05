# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple
import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        # self.set_filename()
        self.action_space = spaces.MultiDiscrete([4])
        """
        Considers a 7*4 gridworld with the following values for states:
        Floor - 0
        Counter - 1
        Cutboard - 2
        Delivery - 3
        Plate - 4
        Tomato - 5
        Agent - 6
        Last 3 states represent:
        Location of tomato
        Location of plate
        State of tomato - 0/1 = Unchopped/Chopped
        """ 
        obs_space = [7 for x in range(28)] + [28, 28, 2]
        self.observation_space = spaces.MultiDiscrete(obs_space)

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model

    def load_level(self, level, num_agents):
        x = 0
        y = 0
        idx = 0
        with open('utils/levels/{}.txt'.format(level), 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlop':
                            counter = Counter(location=(x, y))
                            obj = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                            # only plate and tomato are part of state for now
                            if rep == 'p':
                                self.state[idx] = 4
                            elif rep == 't':
                                self.state[idx] = 5
                                self.tomato_start = idx

                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            if newobj.name == "Counter":
                                self.state[idx] = 1
                            elif newobj.name == "Cutboard":
                                self.state[idx] = 2
                            elif newobj.name == "Delivery":
                                self.state[idx] = 3
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                        idx += 1
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    self.recipes.append(globals()[line]())

                # Phase 3: Read in agent locations (up to num_agents).
                elif phase == 3:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                                name='agent-'+str(len(self.sim_agents)+1),
                                id_color=COLORS[len(self.sim_agents)],
                                location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)
                        idx = (int(loc[1])) * (x+1) + int(loc[0])
                        self.state[idx] = 6
                        

        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)

    # Get index of location in state vector
    def get_state_idx(self, obj):
        return obj.location[1] * self.world.width + obj.location[0]
    
    # Update the state based on current world
    def update_state(self):

        chopped = 0
        tomato_loc = 0
        plate_loc = 0

        # Find out if tomatoes are chopped. Works only when not plated
        for subtask in self.all_subtasks:
            if isinstance(subtask, recipe.Chop):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if goal_obj_locs:
                    chopped = 1

        for obj in self.world.objects["Tomato"]:
            if obj.location:
                tomato_loc = self.get_state_idx(obj)
        objs = []
        for o in self.world.objects.values():
            objs += o

        for obj in objs:
            idx = obj.location[1] * self.world.width + obj.location[0]
            if obj.name == "Counter":
                self.state[idx] = 1
            elif obj.name == "Cutboard":
                self.state[idx] = 2
            elif obj.name == "Delivery":
                self.state[idx] = 3
            elif obj.name == "Plate":
                self.state[idx] = 4
            elif obj.name == "Plate-Tomato":
                tomato_loc = obj.location[1] * self.world.width + obj.location[0]
                plate_loc = obj.location[1] * self.world.width + obj.location[0]
                chopped = 1
                
        self.state[28] = tomato_loc
        self.state[29] = plate_loc
        self.state[30] = chopped

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # State for network
        self.state = np.zeros(31)
        self.pick = False

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)
        self.all_subtasks = self.run_recipes()
        # Keep track of subtask completion
        self.subtask_status = np.zeros(len(self.all_subtasks))
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.record or self.arglist.with_image_obs:
            self.game = GameImage(
                    filename="test",
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record)
            self.game.on_init()
            if self.arglist.record:
                self.game.save_image_obs(self.t)

        return self.state

    def close(self):
        return

    def step(self, action):
        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # Only single agent action
        NAV_ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        action = NAV_ACTIONS[action[0]]

        sim_loc = self.sim_agents[0].location
        sim_state_idx = sim_loc[1] * self.world.width + sim_loc[0]
        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action

        # Check collisions.
        self.check_collisions()
        self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()

        sim_loc = self.sim_agents[0].location
        sim_state_new_idx = sim_loc[1] * self.world.width + sim_loc[0]
        self.state[sim_state_idx] = 0
        self.state[sim_state_new_idx] = 6

        self.update_state()

        # Visualize.
        self.display()
        self.print_agents()
        if self.arglist.record:
            self.game.save_image_obs(self.t)
        # print(self.state)
        # Get a plan-representation observation.
        new_obs = copy.copy(self)
        # Get an image observation
        image_obs = self.game.get_image_obs()

        done = self.done()
        reward = self.reward()
        info = {"t": self.t, "obs": self.state,
                "image_obs": image_obs,
                "done": done, "termination_info": self.termination_info}
        return self.state, reward, done, info


    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True

        assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

        # Done if subtask is completed.
        for subtask in self.all_subtasks:
            # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if not any([gol == delivery_loc for gol in goal_obj_locs]):
                    self.termination_info = ""
                    self.successful = False
                    return False

        self.termination_info = "Terminating because all deliveries were completed"
        self.successful = True
        return True

    def reward(self):
        reward = 0
        for idx, subtask in enumerate(self.all_subtasks):
            # Check if subtask is complete
            if not self.subtask_status[idx]:
                _, goal_obj = nav_utils.get_subtask_obj(subtask)
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if isinstance(subtask, recipe.Deliver):
                    # If task is delivery then give max reward
                    delivery_loc = list(filter(lambda o: o.name=='Delivery',\
                                     self.world.get_object_list()))[0].location
                    if goal_obj_locs:
                        if all([gol == delivery_loc for gol in goal_obj_locs]):
                            reward = 150
                else:
                    # for completion of any other subtasks give small rewards
                    if goal_obj_locs:
                        reward = 20
                        self.subtask_status[idx] = 1
        if not self.tomato_start == self.state[28] and not self.pick:
            reward = 20
            self.pick = True

        if reward == 0:
            reward = -0.01
        print("Reward for step: {}".format(reward))
        return reward

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        # Print states for checking
        state_print = np.copy(self.state)
        state_print_map = state_print[:28].reshape((4,7))
        print(state_print_map)

        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)


    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]
        print('Subtasks:', all_subtasks, '\n')
        return all_subtasks

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances

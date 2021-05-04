# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime


class TestPlay(Game):
    def __init__(self, world, sim_agents, env):
        Game.__init__(self, world, sim_agents, play=True)

        # tally up all gridsquare types
        self.env = env
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # Control current agent
            x, y = self.current_agent.location
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.env.step(action)

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        # while self._running:
        #     for event in pygame.event.get():
        #         self.on_event(event)
        self.on_render()
        # self.on_cleanup()

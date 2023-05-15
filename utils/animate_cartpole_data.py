# This file creates a .gif animation based on a simulation run

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import operator
import json
import os
import tqdm

simulation_file = open('./offline-rl-end-to-end/data/simulation.json')
simulation_dict = json.load(simulation_file)

# x, x_dot, theta, theta_dot = self.state

# Create figure and axes
fig, ax = plt.subplots(1)
ax.set_title('Cartpole Simulation. t=0')

# plt.axes(xlim=(-1., 1.),  ylim=(0., 1.))
plt.xlim(-1., 1.)

xy_cart = (0,0)
base_width = 0.2
base_height = 0.1
arm_width = 0.01
arm_height = 0.2


cartpole_base = Rectangle(xy_cart, base_width, base_height, color='cornflowerblue')  # Rectangle Patch
cartpole_arm = Rectangle(tuple(map(operator.add, xy_cart, (base_width/2-arm_width/2, base_height/2))), arm_width, arm_height)  # Rectangle Patch
goal_location = Rectangle(xy_cart, base_width, base_height/4, color='green')  # Rectangle Patch


ax.add_patch(cartpole_base)  # Add Patch to Plot
ax.add_patch(cartpole_arm)  # Add Patch to Plot
ax.add_patch(goal_location)  # Add Patch to Plot

average_reward = sum([step['reward'] for step in simulation_dict['steps']])/len(simulation_dict['steps'])
action_source = simulation_dict['steps'][0].get('action_source')

# f = fr"./cartpole_avg_rew_{average_reward:.2f}.gif" 
f = r"./cartpole.gif" 
gif_writer = animation.PillowWriter(fps=30) 

with gif_writer.saving(fig, f, dpi = 60):
  for i, step in tqdm.tqdm(enumerate(simulation_dict['steps']), total = len(simulation_dict['steps'])):
    ax.set_title(f'Cartpole Simulation. t={i}, Average Reward: {average_reward:.2f}\n{action_source}')
    # x, x_dot, theta, theta_dot, x_goal = step[['cart_position','cart_velocity','pole_angle','pole_angular_velocity','goal_position']]
    xy_cart = (step['cart_position'], 0)
    
    
    cartpole_base.set(xy=xy_cart)
    cartpole_arm.set(xy=tuple(map(operator.add, xy_cart, (base_width/2-arm_width/2, base_height/2))))
    cartpole_arm.set(angle=-step['pole_angle']*180/np.pi)
    goal_location.set(xy=(step['goal_position'],xy_cart[1]))
    
    # Update Drawing:
    fig.canvas.draw()
    # Save Frame:
    gif_writer.grab_frame()

plt.close()
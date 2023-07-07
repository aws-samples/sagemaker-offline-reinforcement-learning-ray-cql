# This file creates a .gif animation based on a simulation run

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from pathlib import Path
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
import operator
import json
import os
import tqdm

simulation_file = open(os.path.join(Path(__file__).parent.parent.absolute(),'assets','simulation.json'))
simulation_dict = json.load(simulation_file)

# x, x_dot, theta, theta_dot = self.state

# Create figure and axes
fig, ax = plt.subplots(1)
ax.set_title('Cartpole Simulation. t=0')

ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

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
force_arrow = Arrow(xy_cart[0]+base_width/2, base_height/2, 0.1, 0, width=0.01, color='red')  # Arrow describing the force applied to the cartpole

ax.add_patch(cartpole_base)  # Add Patch to Plot
ax.add_patch(cartpole_arm)  # Add Patch to Plot
ax.add_patch(goal_location)  # Add Patch to Plot
ax.add_patch(force_arrow)

average_reward = sum([step['reward'] for step in simulation_dict['steps']])/len(simulation_dict['steps'])
action_source = simulation_dict['steps'][0].get('action_source')

# f = fr"./cartpole_avg_rew_{average_reward:.2f}.gif" 
f = r"./cartpole.gif" 
gif_writer = animation.PillowWriter(fps=30) 

with gif_writer.saving(fig, f, dpi = 60):
  for i, step in tqdm.tqdm(enumerate(simulation_dict['steps']), total = len(simulation_dict['steps'])):
    step['external_force'] = float(step['external_force'])
    
    # ax.set_title(f'Cartpole Simulation. t={i}, Average Reward: {average_reward:.2f}\n{action_source}')
    ax.set_title(f'Training Data', fontsize = 30)
    
    # xy_cart = (step['cart_position'], 0)
    # x, x_dot, theta, theta_dot, x_goal = step[['cart_position','cart_velocity','pole_angle','pole_angular_velocity','goal_position']]
    xy_cart = (step['cart_position'], 0)
    
    
    cartpole_base.set(xy=xy_cart)
    cartpole_arm.set(xy=tuple(map(operator.add, xy_cart, (base_width/2-arm_width/2, base_height/2))))
    cartpole_arm.set(angle=-step['pole_angle']*180/np.pi)
    goal_location.set(xy=(step['goal_position'],xy_cart[1]))
    
    force_arrow.remove() # the force arrow position can not be dynamically updated. It must be removed and re-added.
    force_arrow = Arrow(
      x=xy_cart[0]+base_width/2-np.sign(step['external_force'])*base_width/2-step['external_force'], 
      y=base_height/2, 
      dx=step['external_force'],
      dy=0,
      width = 0.1,
      color='red'
    )
    ax.add_patch(force_arrow)
    
    # Update Drawing:
    fig.canvas.draw()
    # Save Frame:
    gif_writer.grab_frame()

plt.close()
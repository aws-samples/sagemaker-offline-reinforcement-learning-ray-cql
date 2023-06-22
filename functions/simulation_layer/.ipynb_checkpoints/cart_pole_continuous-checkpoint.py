import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

def semi_circle_reward(x,saturated_value,reward_max = 1):
    'This funciton gives maximum reward if x = 0. The reward decreases in the shape of a semi-circle until reward = 0 at abs(x) = saturated_value. For abs(x)>saturated_value, reward = 0.'
    x_normalized = min(abs(x/saturated_value),1)
    return reward_max*math.sqrt(1-x_normalized**2)

class ContinuousCartPoleEnv(gym.Env):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        
        self.x_goal = 0.0 #Initial goal location.
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot, x_goal = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        return (x, x_dot, theta, theta_dot, self.x_goal)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot, x_goal = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)
        
        reward = (abs(x-x_goal)<0.1)
        # reward = 0
        # # reward += semi_circle_reward((x-x_goal),saturated_value = 0.5, reward_max = 10) #Give a reward of 10 if x=x_goal. No rewards if abs(x-x_goal)> 0.5
        # reward += (abs(x-x_goal)<0.1)*10
        # reward += semi_circle_reward(x_dot,     saturated_value = 0.5, reward_max = 1)
        # reward += semi_circle_reward(theta,     saturated_value = 0.5, reward_max = 1)
        # reward += semi_circle_reward(theta_dot, saturated_value = 0.5, reward_max = 1)
            
        if self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        self.steps_beyond_done = None
        return np.array(self.state)
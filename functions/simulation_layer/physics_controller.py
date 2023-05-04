#Build by waltmayf@ with help from this blog: https://towardsdatascience.com/comparing-optimal-control-and-reinforcement-learning-using-the-cart-pole-swing-up-openai-gym-772636bc48f4

import numpy as np
from scipy import linalg

class PhysicsController:
  def __init__(self, mk, mp, lp): # lp = length pendulum, mp = mass pendulum, mk = mass cart
    g = 9.81 #m/s2
    
    # state matrix
    a = g/(lp*(4.0/3 - mp/(mp+mk)))
    A = np.array([    # How the state would change without any action taken
        [0, 1, 0, 0], #s dot
        [0, 0, a, 0], #s dot dot
        [0, 0, 0, 1], #theta dot
        [0, 0, a, 0]  #theta dot dot
      ])
    
    # input matrix
    b = -1/(lp*(4.0/3 - mp/(mp+mk)))
    B = np.array([[0], [1/(mp+mk)], [0], [b]]) #How a given action effects the state
    
    # Controller Inputs
    R = 1*np.eye(1, dtype=int)          # choose R (weight for input) 
    # Q = 100*np.eye(4, dtype=int)        # choose Q (weight for state)
    Q = np.zeros((4,4))
    Q[0,0] = 100 #Reduce the position of the cart to the origin
    Q[2,2] = 1 #Reduce the angle of the pole to upright
    
    # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)
    
    # calculate optimal controller gain
    self.K = np.dot(np.linalg.inv(R),
                    np.dot(B.T, P))
  
  def apply_state_controller(self, obs):
    # feedback controller
    u = -np.dot(self.K, obs)   # u = -Kx
    return u


if __name__ == '__main__':
  test_controller = PhysicsController(mk=1, mp=1, lp=1)
  print(test_controller.K)


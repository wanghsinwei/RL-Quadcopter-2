import numpy as np
from .physics_sim import PhysicsSim

class VerticalTakeoffTask:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5.):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 1

        self.state_size = self.action_repeat * 9 # 3 Euler angles + 3 velocity + 3 angle velocity
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4 # 4 rotors

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        
        # if dist < self.last_dist:
        #     reward = (self.last_dist - dist)
        # else:
        #     reward = 2*(self.last_dist - dist)
        # self.last_dist = dist

        # if self.sim.pose[2] < self.target_pos[2]:
        #     if self.sim.pose[2] > self.last_pos[2]:
        #         reward += (self.sim.pose[2] - self.last_pos[2])
        #     else:
        #         reward += 2*(self.sim.pose[2] - self.last_pos[2])
            
        # self.last_pos = self.sim.pose[:3]

        # if self.sim.pose[2] > 0:
        #     reward += 0.1

        # reward = 0

        reward = 0.8*self.sim.v[2] - 0.2*(abs(self.sim.v[0]) + abs(self.sim.v[1])) - 0.01*np.abs(self.sim.angular_v).sum()

        # forward_vec = self.target_pos - self.sim.pose[:3]
        # forward_velocity = np.inner(self.sim.v, forward_vec) / np.linalg.norm(forward_vec)
        # if forward_velocity > 0:
        #     reward += 5*forward_velocity
        # # else:
        # #     reward += forward_velocity

        # deviation_velocity = np.linalg.norm(self.sim.v) - abs(forward_velocity)
        # reward -= .1*deviation_velocity

        # if dist <= 0.5:
        #     reward = 50

        # Crash Penalty
        if self.sim.done and self.sim.time <= self.sim.runtime:
            reward = -5

        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            state_all.append(self.sim.pose[3:])
            state_all.append(self.sim.v)
            state_all.append(self.sim.angular_v)
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose[3:], self.sim.v, self.sim.angular_v] * self.action_repeat)
        return state
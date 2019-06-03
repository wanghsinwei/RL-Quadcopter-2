import numpy as np
from .physics_sim import PhysicsSim

class ReachTargetTask:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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

        self.state_size = self.action_repeat * 13 # 3 Euler angles + 3 velocity + 3 angle velocity + target vector + dist
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4 # 4 rotors

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.success = False
        self.goal_dist = 0.5

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        target_vec = self.target_pos - self.sim.pose[:3]
        dist = np.linalg.norm(target_vec)
        target_unit_vec = target_vec / dist
        v_proj_target = np.inner(self.sim.v, target_unit_vec) * target_unit_vec

        reward = 0.1*v_proj_target[2] + 0.01*(v_proj_target[0] + v_proj_target[1])

        # Reach Goal
        if self.success == True:
            reward = 50

        # Crash Penalty
        if self.sim.done and self.sim.pose[2] == 0:
            reward = -10

        return np.tanh(reward)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            target_vec = self.target_pos - self.sim.pose[:3]
            dist = np.linalg.norm(target_vec)
            if not done and dist <= self.goal_dist:
                done = True
                self.success = True
            reward += self.get_reward() 
            state_all.append(self.sim.pose[3:])
            state_all.append(self.sim.v)
            state_all.append(self.sim.angular_v)
            state_all.append(target_vec)
            state_all.append(np.array([dist]))
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        target_vec = self.target_pos - self.sim.pose[:3]
        dist = np.linalg.norm(target_vec)
        state = np.concatenate([self.sim.pose[3:], self.sim.v, self.sim.angular_v, target_vec, np.array([dist])] * self.action_repeat)
        self.success = False
        return state

    def sample_act(self, action_low, action_high):
        return np.random.uniform(action_low, action_high, self.action_size)
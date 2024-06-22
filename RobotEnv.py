
import numpy as np
from gym import utils
from collections import deque
import math
from scipy.spatial.transform import Rotation
from gym.envs.mujoco import MujocoEnv
from GaitController import GaitController
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 20.0)),
    "elevation": 20.0,
}

class RobotEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
        }

    def __init__(self, render=True, swingFootHeight = 0.05): 
        self.is_render = render

        # Gait Controller Parameters
        self.cStp = 0.008
        self.dStp = 0.001
        self.omega = 2 * np.pi 
        phase = [0, np.pi] 
        self.gaitCon = GaitController(phase,r_leg_indx=[1,2,3],l_leg_indx=[4,5,6],r_foot_site_indx=0,l_foot_site_indx=1)
        self.gaitCon.foot_clearance = swingFootHeight
        self.last_base_position = [0, 0, 0]
        self.stance = None
        # Learning parameters
        self.obs_dim = 6 # torso rpy and angular rates
        self.action_dim = 4 # steplength, shifts in x, y, z
        observation_space = Box(low=-np.inf,high=np.inf,shape=(self.obs_dim,),dtype=np.float64)
        MujocoEnv.__init__(self, 'biped.xml', 1,observation_space)

        # Environment parameters
        self.n_env_step = 0

     ###############    ###############
    def transform_gym_action_to_mujoco(self,action):
        # gym action is 4x1 vector
        action_dummy = np.array([action[0],action[1],action[2],action[3],action[4]])
        action_mujoco = [action_dummy[0],action_dummy[0],
                        action_dummy[1],action_dummy[1],
                        action_dummy[2],action_dummy[2],
                        action_dummy[3],action_dummy[3],
                        action_dummy[4],action_dummy[4]]
        return action_mujoco
    def step(self, action):

        X, Y, Z , ang_commands = self.gaitCon.generate_elliptical_trajectory(curr_phase=self.phase,
                                                                      action=action)

        joint_ang_commands = np.array(ang_commands,dtype=list)
        joint_vel_commands = np.zeros(len(ang_commands))
        torque_commands = self.pd_control(joint_ang_commands, joint_vel_commands) # check proper implementation
        torque_commands = np.clip(torque_commands,-10, 10)
        n_frames = int(self.cStp / self.dt)

        self.do_simulation(torque_commands, n_frames) # check proper implementation
        self.n_env_step +=1

        if(self.is_render):
            self.render_mode = "human"
            self.render()

        observation = self._get_obs()
        self.phase =  self.update_phase(self.omega * self.cStp + self.phase)
        # check for stance of robot -- update self.stance 
        reward,done,reward_list = self._get_reward()

        return observation,reward, done, {}
    
    def _chk_termination(self,pos,base_rpy):

        done = False
        height = pos[2]
        s = self.state_vector()
        torso_pos,_ = self._get_base_pos_ori() 
       
        if not np.isfinite(s).all():
            done = True
        if height < 0.7*self.desired_height or height > 2*self.desired_height:
            done = True

        if True in (np.abs(base_rpy[0:2]) > np.array([20,20],dtype=list)*180/np.pi) :    
            done = True
            
        return done 

    def _get_reward(self):

        # get current values
        pos, ori = self._get_base_pos_ori()
        base_rpy = self.quaternion_to_euler_angle(ori)
        lin_vel, ang_vel = self._get_base_vel_omega()

        rot_world_to_base = np.asarray(self.euler_angles_rotation_matrix(base_rpy))
        lin_vel_body = np.dot(rot_world_to_base,lin_vel)

        base_rpy = np.round(base_rpy, 4)
        current_height = round(pos[2], 4)

        # messure displacement in this time step
        x = pos[0]
        x_l = self.last_base_position[0]
        self.last_base_position = list(pos)

        # Desired targets
        torso_roll_target = np.radians(0)
        torso_pitch_target = np.radians(0)

        
        self.desired_height = 0.875
        self.des_heading_vel = [0.1,0] 

        # Reward Terms
        roll_reward = np.exp(-250 * ((base_rpy[0]-torso_roll_target) ** 2)) 
        pitch_reward = np.exp(-400 * ((base_rpy[1]-torso_pitch_target) ** 2)) 

        vel_reward_x = np.exp(-10 * ((lin_vel[0] - self.des_heading_vel[0]) ** 2))
        vel_reward_y = np.exp(-100 * ((lin_vel[1]) ** 2))

        height_reward = np.exp(-500 * (self.desired_height - current_height) ** 2) #1000
        step_distance_x = (x - x_l)
        step_reward = 300*step_distance_x
        
        reward_list = [
                        round(roll_reward,4), 
                        round(pitch_reward, 4), 
                        round(height_reward,4), 
                        round(vel_reward_x,4),
                        round(vel_reward_y,4),
                        round(step_reward, 4),
                        ]           

        done = self._chk_termination(pos, base_rpy)
        if done:
            reward = 0
        else:
            reward = sum(reward_list)

        return reward, done,reward_list
    
    def _get_obs(self):

        base_pos,base_orientation_q = self._get_base_pos_ori() 
        base_rpy = self.quaternion_to_euler_angle(base_orientation_q)  
        
        lin_vel,ang_vel = self._get_base_vel_omega() 
        R_from_world_to_base = np.asarray(self.euler_angles_rotation_matrix(base_rpy)) # try to get from mujoco, it is simply rotation matrix from euler angles
        lin_vel_body = np.dot(R_from_world_to_base,lin_vel)        
        ang_vel = list(ang_vel)

        obs = np.concatenate((
                              base_rpy,
                              ang_vel)).ravel()

        return obs

    def _get_base_pos_ori(self):
        return self.data.qpos[0:3],self.data.qpos[3:7]

    def _get_base_vel_omega(self):
        return self.data.qvel[0:3],self.data.qvel[3:6]   

    def reset_model(self):

        self.n_env_step = 0

        initial_phase = np.random.choice(np.radians([90]))
        self.phase = initial_phase
        
        initial_leg_action = np.zeros(8)
        # actions --> motor angles
        X, Y, Z , ang_commands = self.gaitCon.generate_elliptical_trajectory(self.phase,initial_leg_action)
        
        init_pos = np.array([0.0]*len(self.data.qpos),dtype=list)
        init_vel = np.array([0.0]*len(self.data.qvel),dtype=list)
        
        qpos = init_pos
        qvel = init_vel
        self.set_state(qpos, qvel)

        if initial_phase >= np.radians(0) and initial_phase < np.radians(180) :
            self.stance = "left"
        else:
            self.stance = "right"
        return self._get_obs()

    def _get_motor_pos_vel(self):
        return [self.data.qpos[i] for i in [7,8,9,10,11,12]], [self.data.qvel[i] for i in [6,7,8,9,10,11]]


    def pd_control(self,qpos_ref, qvel_ref,kp=[170,170,100,170,170,100],kd=[20,8,5,20,8,5]):
 
        qpos_act, qvel_act  = self._get_motor_pos_vel()
        state = np.concatenate((self.data.qpos, self.data.qvel)).ravel()

        ctrl = np.ones(6)
        for i in range(6):
            ctrl[i] = -kp[i]*(qpos_act[i]-qpos_ref[i]) - kd[i]*(qvel_act[i] - qvel_ref[i])     
        applied_motor_torque = ctrl.tolist()
        return applied_motor_torque

    def update_phase(self,theta):
        if(theta > 2*np.pi):
            theta = 0.0
        return theta
    def quaternion_to_euler_angle(self,q):

        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.atan2(t3, t4)

        return [X, Y, Z]
    def euler_angles_rotation_matrix(self,rpy):

        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
        ])

        pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
        ])

        R = yawMatrix * pitchMatrix * rollMatrix

        return R
if __name__ == "__main__":
    env = RobotEnv()
    env.reset()
    render_mode='human'
    env.render()
    env.step(np.zeros(8))
import math
import numpy as np
import os
import mujoco as mj
from utils import quaternion_to_euler_angle, euler_angles_rotation_matrix

class InverseKinematics:
    def __init__(self,r_leg_indx=[1,2,3],l_leg_indx=[4,5,6],r_foot_site_indx=0,l_foot_site_indx=1):
        #pass
        self.r_leg_indx=r_leg_indx
        self.l_leg_indx=l_leg_indx
        self.r_foot_indx = r_foot_site_indx
        self.l_foot_indx = l_foot_site_indx
        
        # xml_path = 'biped_pdtuning.xml'
        # dirname = os.path.dirname(__file__)
        # abspath = os.path.join(dirname + "/" + xml_path)
        # xml_path = abspath

        # # MuJoCo data structures
        # self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        # self.data = mj.MjData(self.model)

        # theta1  = 0
        # theta2  = 0
        # theta3  = 0
        # theta4  = 0
        # theta5  = 0
        # theta6  = 0

        # #initialize
        # self.data.qpos[0] = theta1
        # self.data.qpos[1] = theta2
        # self.data.qpos[2] = theta3
        # self.data.qpos[3] = theta4
        # self.data.qpos[4] = theta5
        # self.data.qpos[5] = theta6

        # mj.mj_forward(self.model,self.data)
        
    def get(self,X_,Y_,Z_,model,data,flag): # to be used with RobotEnv
        base_pos,base_orientation_q = data.qpos[0:3],data.qpos[3:7]
        base_rpy = self.quaternion_to_euler_angle(base_orientation_q)  
        
        R_from_world_to_base = np.asarray(self.euler_angles_rotation_matrix(base_rpy)) # try to get from mujoco, it is simply rotation matrix from euler angles
        rpos = np.array([X_,Y_,Z_])
        rpos_world = np.dot(R_from_world_to_base.T,rpos)
        #Z = Z+1.2
        X = rpos_world[0]
        Y = rpos_world[1]
        Z = rpos_world[2]
        mj.mj_forward(model,data)
        if flag == "r":
            #position_Q = self.data.site_xpos[self.r_foot_indx]
            position_Q = self._get_ref_coordinates(model,data,flag)
            jacp = np.zeros((3,6)) #3 is for x,y,z and 2 is for theta1 and theta2
            mj.mj_jac(model,data,jacp,None,position_Q,4)
            # print(jacp)
            J = jacp[:,0:3]
            # J = jacp
            Jinv = np.linalg.pinv(J)
            dX = np.array([X-position_Q[0],Y- position_Q[1],Z- position_Q[2]])
            dq = Jinv.dot(dX)
        elif flag == "l":
            #position_Q = self.data.site_xpos[self.l_leg_indx]
            position_Q = self._get_ref_coordinates(model,data,flag)
            # print(position_Q)
            jacp = np.zeros((3,6)) #3 is for x,y,z and 2 is for theta1 and theta2
            mj.mj_jac(model,data,jacp,None,position_Q,7)
            J = jacp[:,3:6]
            Jinv = np.linalg.pinv(J)
            dX = np.array([X-position_Q[0],Y- position_Q[1],Z- position_Q[2]])
            dq = Jinv.dot(dX)
        else:
            print("Leg index error")
        return dq
    
    def __call__(self,X,Y,Z,model,data,flag): # to be used in test_gait_controller
        Z = Z+1.2
        mj.mj_forward(model,data)
        if flag == "r":
            #position_Q = self.data.site_xpos[self.r_foot_indx]
            position_Q = self._get_ref_coordinates(model,data,flag)
            jacp = np.zeros((3,6)) #3 is for x,y,z and 2 is for theta1 and theta2
            mj.mj_jac(model,data,jacp,None,position_Q,4)
            # print(jacp)
            J = jacp[:,0:3]
            # J = jacp
            Jinv = np.linalg.pinv(J)
            dX = np.array([X-position_Q[0],Y- position_Q[1],Z- position_Q[2]])
            dq = Jinv.dot(dX)
        elif flag == "l":
            #position_Q = self.data.site_xpos[self.l_leg_indx]
            position_Q = self._get_ref_coordinates(model,data,flag)
            # print(position_Q)
            jacp = np.zeros((3,6)) #3 is for x,y,z and 2 is for theta1 and theta2
            mj.mj_jac(model,data,jacp,None,position_Q,7)
            J = jacp[:,3:6]
            Jinv = np.linalg.pinv(J)
            dX = np.array([X-position_Q[0],Y- position_Q[1],Z- position_Q[2]])
            dq = Jinv.dot(dX)
        else:
            print("Leg index error")
        return dq
    def _get_ref_coordinates(self,model,data,flag='r'):
        data_buffer = []
        if (flag == 'r'):
            data_buffer = [data.qpos[i] for i in self.r_leg_indx]
            #data_buffer[0] = data.qpos[self.r_foot_indx[0]]
            #data_buffer[1] = data.qpos[self.r_foot_indx[1]]
            #data_buffer[2] = data.qpos[self.r_foot_indx[2]]
            data.qpos[self.r_leg_indx[0]] = 0
            data.qpos[self.r_leg_indx[1]] = 0
            data.qpos[self.r_leg_indx[2]] = 0
            mj.mj_forward(model,data)
            position_Q = data.site_xpos[self.r_foot_indx]
            data.qpos[self.r_leg_indx[0]] = data_buffer[0]
            data.qpos[self.r_leg_indx[1]] = data_buffer[1]
            data.qpos[self.r_leg_indx[2]] = data_buffer[2] 
            return position_Q       
        else:
            data_buffer = [data.qpos[i] for i in self.l_leg_indx]
            #data_buffer[0] = data.qpos[self.l_foot_indx[0]]
            #data_buffer[1] = data.qpos[self.l_foot_indx[1]]
            #data_buffer[2] = data.qpos[self.l_foot_indx[2]]
            data.qpos[self.l_leg_indx[0]] = 0
            data.qpos[self.l_leg_indx[1]] = 0
            data.qpos[self.l_leg_indx[2]] = 0
            mj.mj_forward(model,data)
            position_Q = data.site_xpos[self.l_foot_indx]
            data.qpos[self.l_leg_indx[0]] = data_buffer[0]
            data.qpos[self.l_leg_indx[1]] = data_buffer[1]
            data.qpos[self.l_leg_indx[2]] = data_buffer[2] 
            return position_Q 
        
                
    
if __name__ == "__main__":
    ik = InverseKinematics()
    # print(ik(0.255, -0.15632468, -0.83695862))
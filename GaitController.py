from dataclasses import dataclass
from collections import namedtuple
import numpy as np
import math
from Kinematics import InverseKinematics


@dataclass
class leg_data:
    name: str
    hip_angle: float = 0.0    
    thigh_angle: float = 0.0
    shank_angle: float = 0.0
    foot_x: float = 0.0
    foot_y: float = 0.0
    foot_z: float = 0.0
    phase_angle: float = 0.0
    steering_angle: float = 0.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0


@dataclass
class robot_data:
    right: leg_data = leg_data('r')
    left: leg_data = leg_data('l')

class GaitController():
    def __init__(self,phase=[0, 0],r_leg_indx=[0,1,2],l_leg_indx=[0,1,2],r_foot_site_indx=0,l_foot_site_indx=1):
        self._phase = robot_data(right=phase[0], left=phase[1])
        self.right = leg_data('r')
        self.left = leg_data('l')

        self.z_center = -0.9  # fill in the height of the robot (from torso center to foot) at zero configuration
        self.foot_clearance = 0.05 
        self.ik = InverseKinematics(r_leg_indx,l_leg_indx,r_foot_site_indx,l_foot_site_indx)

    
    def update_leg_phase_angle(self, phase):

        self.right.phase_angle = np.fmod(phase + self._phase.right, 2*np.pi)
        self.left.phase_angle  = np.fmod(phase + self._phase.left, 2*np.pi)
      

    def initialize_elipse_shift(self, Xshift,Yshift, Zshift):
        
        self.right.x_shift = Xshift[0]
        self.left.x_shift = Xshift[1]

        self.right.y_shift = Yshift[0]
        self.left.y_shift = Yshift[1]
 
        self.right.z_shift = Zshift[0]
        self.left.z_shift = Zshift[1]


    def initialize_leg_state(self, curr_phase, action):
        
        Legs = namedtuple('legs', 'right left')
        legs = Legs(right=self.right, left=self.left)

        self.update_leg_phase_angle(curr_phase)

        leg_sl = action[0:2]  # R, L
        # leg_steering = action[2:4]  # R, L

        # self._update_leg_steering(leg_steering)
        self._update_leg_step_length_val(leg_sl)

        self.initialize_elipse_shift(action[2:4], action[4:6], action[6:8])
        return legs

    def generate_elliptical_trajectory(self, curr_phase, action,model,data):

        legs = self.initialize_leg_state(curr_phase, action)

        valid_list = []
        X = []
        Y = []
        Z = []
        joint_angles_cmd  =[]
        for leg in legs:
            
            x = -0.5*leg.step_length * np.cos(leg.phase_angle) 
            if leg.phase_angle > np.pi:
                flag = 0
            else:
                flag = 1
            z = self.foot_clearance * np.sin(leg.phase_angle) * flag + self.z_center #+ leg.z_shift
            
            leg.foot_x = x
            if leg.name=="r":
                leg.foot_y = -0.17292
            else :
                leg.foot_y = -0.17292
            leg.foot_z = z
            
            leg.foot_x = leg.foot_x + leg.x_shift
            leg.foot_y = leg.foot_y + leg.y_shift
            leg.foot_z = leg.foot_z + leg.z_shift
        
            

            X.append(leg.foot_x)
            Y.append(leg.foot_y)
            Z.append(leg.foot_z)

           # Call IK here
            joint_angles_cmd.append(self.ik(leg.foot_x,leg.foot_y,leg.foot_z,model,data,leg.name))
        joint_angles_cmd_flatten = [element for sub_list in joint_angles_cmd for element in sub_list]  
        return X,Y,Z, joint_angles_cmd_flatten
    def _update_leg_steering(self, steering_angle):

        self.right.steering_angle = steering_angle[0]  
        self.left.steering_angle = steering_angle[1] 


    def _update_leg_step_length_val(self, step_length):
        
        self.right.step_length = step_length[0]
        self.left.step_length = step_length[1]

if (__name__ == "__main__"):
    gc = GaitController([0,np.pi])
    action_dummy = np.array([0.25,0,0,0,0])
    curr_phase = np.pi
    tau = 0
    n_step = 1
    dt = 0.001
    w = 2*np.pi
    N = 1/dt
    i = 0
    l_pos = []
    r_pos = []
    time = []
    ctime = 0
    cstep = 0
    phase = []
    while(cstep<= n_step):
        action = [action_dummy[0],action_dummy[0],
                  action_dummy[1],action_dummy[1],
                  action_dummy[2],action_dummy[2],
                  action_dummy[3],action_dummy[3],
                  action_dummy[4],action_dummy[4]]
        [X,Y,Z,dummy] = gc.generate_elliptical_trajectory(curr_phase, action)
        l_pos.append([X[1],Y[1],Z[1]])
        r_pos.append([X[0],Y[0],Z[0]])
        i = i+1
        curr_phase = curr_phase + w*dt
        if (curr_phase >= 2*np.pi):
            curr_phase = 0
        ctime = ctime+dt
        time.append(ctime)
        phase.append(curr_phase)
        if (i>=N):
            i=0
            #action_dummy[2]+= 0.25
            cstep = cstep+1
            curr_phase = np.pi
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    l_pos = np.array(l_pos)
    # print(l_pos.shape)
    r_pos = np.array(r_pos)
    X = l_pos[:,0]
    Y = l_pos[:,1]
    Z = l_pos[:,2]
    X_d = r_pos[:,0]
    Y_d = r_pos[:,1]
    Z_d = r_pos[:,2]
    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.scatter(X, Y, Z, c='r', marker='o', label='Data 1')
    ax.scatter(X_d, Y_d, Z_d, c='b', marker='^', label='Data 2')
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface Plot')
    fig = plt.figure()
    plt.plot(time,X,c='b')
    plt.plot(time,Z,c='b')
    #fig = plt.figure()
    plt.plot(time,X_d,c='r')
    plt.plot(time,Z_d,c='r')
    # Show the plot
    plt.plot(time,phase)

    plt.show()
        
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
from GaitController import GaitController

def rad2deg(ang):
    return ang*180/math.pi
def deg2rad(ang):
    return ang*math.pi/180

xml_path = 'biped.xml' #xml file (assumes this is in the same folder as this file)
simend = 1 #simulation time
print_camera_config = 0 #set to 1 to print camera config

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    pass

def controller(pos_ref,vel_ref, kpVec, kdVec):
    set_torque_servo(0, 1)
    set_torque_servo(1, 1)
    set_torque_servo(2, 1)
    ctrl = np.zeros(6)
    for i in range(6):
        ctrl[i] = -kpVec[i]*(data.qpos[i]-pos_ref[i]) -kdVec[i]*(data.qvel[i] - vel_ref[i]) 
    return ctrl

def set_torque_servo(actuator_no, flag):
    if (flag==0):
        model.actuator_gainprm[actuator_no, 0] = 0
    else:
        model.actuator_gainprm[actuator_no, 0] = 1
    

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
# theta1  = 0
# theta2  = 0
# theta3  = 0

# #initialize
# data.qpos[0] = theta1
# data.qpos[1] = theta2
# data.qpos[2] = theta3
mj.mj_forward(model,data)
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
#initialize the controller here. This function is called once, in the beginning
cam.azimuth = 89.83044433593757 ; cam.elevation = -89.0 ; cam.distance =  5.04038754800176
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

#initialize the controller
init_controller(model,data)

i = 0
time = 0
dt = 0.001
cstp = 0.008
kpVec = [170,170,100,170,170,100]
kdVec = [20,8,5,20,8,5]
gc = GaitController([0,np.pi])
action_dummy = np.array([0.1,0,0,0,0])
curr_phase = np.pi
tau = 0
n_step = 1
w = 2*np.pi
N = 1/dt
i = 0
l_pos = []
r_pos = []
phase = []
hip_q_pos = []
thigh_q_pos = []
shank_q_pos = []
hip_q_pos_r = []
thigh_q_pos_r = []
shank_q_pos_r = []
hip_q_pos_l = []
thigh_q_pos_l = []
shank_q_pos_l = []
time_vec = []
phase_init = curr_phase
while not glfw.window_should_close(window):
    time_prev = time
    action = [action_dummy[0],action_dummy[0],
                  action_dummy[1],action_dummy[1],
                  action_dummy[2],action_dummy[2],
                  action_dummy[3],action_dummy[3],
                  action_dummy[4],action_dummy[4]]
    [X,Y,Z,joint_angle_cmd] = gc.generate_elliptical_trajectory(curr_phase, action,model,data)
    l_pos.append([X[1],Y[1],Z[1]])
    r_pos.append([X[0],Y[0],Z[0]])
    #i = i+1
    curr_phase = curr_phase + w*cstp
    if (curr_phase - phase_init >= 2*np.pi):
        curr_phase = np.pi

    #if (curr_phase >= 2*np.pi):
    #    curr_phase = 0
    phase.append(curr_phase)
    pos_ref = [joint_angle_cmd[0],joint_angle_cmd[1],joint_angle_cmd[2],joint_angle_cmd[3],joint_angle_cmd[4],joint_angle_cmd[5]]
    hip_q_pos_r.append(joint_angle_cmd[0]*180/np.pi)
    thigh_q_pos_r.append(joint_angle_cmd[1]*180/np.pi)
    shank_q_pos_r.append(joint_angle_cmd[2]*180/np.pi)
    # hip_q_pos_l.append(joint_angle_cmd[3])
    # thigh_q_pos_l.append(joint_angle_cmd[4])
    # shank_q_pos_l.append(joint_angle_cmd[5])

    vel_ref = [0,0,0,0,0,0]
    ctrl = controller(pos_ref,vel_ref,kpVec,kdVec)
    data.ctrl[:] = ctrl
    while (time - time_prev < cstp):
        time +=dt
        mj.mj_step(model, data)

    i +=1
    hip_q_pos.append(data.qpos[0]*180/np.pi)
    thigh_q_pos.append(data.qpos[1]*180/np.pi)
    shank_q_pos.append(data.qpos[2]*180/np.pi)
    time_vec.append(time)
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
fig = plt.figure()
plt.plot(time_vec,hip_q_pos,c='b')
plt.plot(time_vec,hip_q_pos_r,c='r')
plt.title('hip joint tracking')
plt.xlabel('Time[s]');plt.ylabel('Hip joint angle')
fig = plt.figure()
plt.plot(time_vec,thigh_q_pos,c='b')
plt.plot(time_vec,thigh_q_pos_r,c='r')
plt.title('thigh joint tracking')
plt.xlabel('Time[s]');plt.ylabel('Thigh joint angle')
fig = plt.figure()
plt.plot(time_vec,shank_q_pos,c='b')
plt.plot(time_vec,shank_q_pos_r,c='r')
plt.title('shank joint tracking')
plt.xlabel('Time[s]');plt.ylabel('Shank joint angle')

plt.show()
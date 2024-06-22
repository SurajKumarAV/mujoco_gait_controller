import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt

def rad2deg(ang):
    return ang*180/math.pi
def deg2rad(ang):
    return ang*math.pi/180







xml_path = 'biped_pdtuning.xml' #xml file (assumes this is in the same folder as this file)
simend = 1 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

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
    ctrl = np.zeros(3)
    for i in range(3):
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

#set the controller
#mj.set_mjcb_control(controller)

# #initialize
# ang1 = deg2rad(90)
# ang2 = deg2rad(90)
# ang3 = deg2rad(90+25)
# data.qpos[1] = ang1
# data.qpos[2] = ang2
# data.qpos[3] = ang3
# mj.mj_forward(model,data)

i = 0
time = 0
dt = 0.001
cstp = 0.008

pos_ref = [np.pi/3,np.pi/10,np.pi/10,0,0,0]
vel_ref = [0,0,0,0,0,0]
kpVec = [100,170,100,0,0,0]
kdVec = [20,8,5,0,0,0]
hip_q_pos = []
thigh_q_pos = []
shank_q_pos = []
time_vec = []
while not glfw.window_should_close(window):
    time_prev = time
    ctrl = controller(pos_ref,vel_ref,kpVec,kdVec)
    data.ctrl[:] = ctrl
    while (time - time_prev < cstp):
        time +=dt
        mj.mj_step(model, data)

    i +=1
    hip_q_pos.append(data.qpos[0])
    thigh_q_pos.append(data.qpos[1])
    shank_q_pos.append(data.qpos[2])
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
plt.title('hip')
fig = plt.figure()
plt.plot(time_vec,thigh_q_pos,c='b')
plt.title('thigh')
fig = plt.figure()
plt.plot(time_vec,shank_q_pos,c='b')
plt.title('shank')
plt.show()
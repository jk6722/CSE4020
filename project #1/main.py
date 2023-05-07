from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import prepare_vao as pv

g_cam_ang = 0.
g_cam_height = 0.
trans_x = 0.
trans_y = 0.

prev_xpos = 0.
prev_ypos = 0.
init_xpos = 0.
init_ypos = 0.

cam_distance = 5.0
flip_up_vector = 1.0
left_click = False
right_click = False

use_ortho = False



g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def button_callback(window, button, action, mod):
    global left_click, right_click, prev_xpos, prev_ypos, init_xpos, init_ypos
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            left_click = True
            prev_xpos, prev_ypos = glfwGetCursorPos(window)
        elif action==GLFW_RELEASE:
            left_click = False
    if button==GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            right_click = True
            init_xpos, init_ypos = glfwGetCursorPos(window)
        elif action==GLFW_RELEASE:
            right_click = False
        

def scroll_callback(window, xoffset, yoffset):
    global cam_distance
    if 1.0 <= cam_distance and cam_distance <= 15.0:
        cam_distance -= yoffset
    if cam_distance <= 1.0:
        cam_distance = 1.0
    if cam_distance >= 15.0:
        cam_distance = 15.0

def cursor_callback(window, xpos, ypos):
    global right_click, left_click, g_cam_ang, g_cam_height, trans_x, trans_y, prev_xpos, prev_ypos, init_xpos, init_ypos,flip_up_vector

    if right_click == True:
        trans_x += (init_xpos - xpos) * .00002
        trans_y -= (init_ypos - ypos) * .00002
    if left_click == True:
        prev_ang = g_cam_height
        g_cam_ang += np.radians((prev_xpos - xpos) * .1)
        g_cam_height += np.radians((prev_ypos - ypos) * .2)

        if(np.cos(prev_ang) * np.cos(g_cam_height) < 0):
            g_cam_ang += np.radians(180)
            flip_up_vector *= -1
        prev_xpos = xpos
        prev_ypos = ypos

def key_callback(window, key, scancode, action, mods):
    global use_ortho
    if action==GLFW_PRESS or action==GLFW_REPEAT:
        if key==GLFW_KEY_V:
            if(use_ortho == True):
                use_ortho = False
            else:
                use_ortho = True

def main():
    global target
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2019049716_안재국', None, None)

    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetKeyCallback(window, key_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = pv.prepare_vao_frame()
    vao_row_grid = pv.prepare_vao_row_grid()
    vao_col_grid = pv.prepare_vao_col_grid()
    vao_cube = pv.prepare_vao_cube()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        # use orthogonal projection (we'll see details later)
        if(use_ortho):
            P = glm.ortho(-cam_distance * 2,cam_distance * 2, -cam_distance * 2,cam_distance * 2,-10,10)
            # P = glm.ortho(-5,5,-5,5,-1,10)
        else: P = glm.perspective(45, 800/800, 1, 100)


        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        
        cam_pos=glm.vec3(
            cam_distance*np.cos(g_cam_height)*np.sin(g_cam_ang),
            cam_distance * np.sin(g_cam_height),
            cam_distance*np.cos(g_cam_height)*np.cos(g_cam_ang)
        )

        V_orbit = glm.lookAt(glm.vec3(flip_up_vector, 1, flip_up_vector) * cam_pos, glm.vec3(0,0,0), glm.vec3(0,1,0) * flip_up_vector)
        V_pan = glm.lookAt(glm.vec3(trans_x,trans_y, 0), glm.vec3(trans_x, trans_y, -1), glm.vec3(0,1,0))
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V_pan*V_orbit*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame) #global frame
        glDrawArrays(GL_LINES, 0, 6)

        glBindVertexArray(vao_cube)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        for i in range (0,50):
            T = glm.translate(glm.vec3(0, 0, i))
            M = T
            MVP = P*V_pan*V_orbit*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_row_grid)
            glDrawArrays(GL_LINES, 0, 6)

            T = glm.translate(glm.vec3(0, 0, -i))
            M = T
            MVP = P*V_pan*V_orbit*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_row_grid)
            glDrawArrays(GL_LINES, 0, 6)

        for i in range (0, 50):
            T = glm.translate(glm.vec3(i, 0, 0))
            M = T
            MVP = P*V_pan*V_orbit*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_col_grid)
            glDrawArrays(GL_LINES, 0, 6)

            T = glm.translate(glm.vec3(-i, 0, 0))
            M = T
            MVP = P*V_pan*V_orbit*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_col_grid)
            glDrawArrays(GL_LINES, 0, 6)
        

        # current frame: P*V*M
        # glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()

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
cam_distance = 7.0
flip_up_vector = 1.0
left_click = False
right_click = False
vao_temp = -1

use_ortho = False

Root = 0
channelList = {}
parent = {}
offset = {}
frameorder = []
allElements = []
use_ortho = False
Nodes_one = {}
Nodes_two = {}
Frames = 0
FrameTime = 0
frame_list_idx = 0

frame_list = []
vao_list_line = []
vao_list_box = []
g_P = 0
motion_mode = False
is_scaled = False

key_one = True
key_two = False


g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize(mat3(inverse(transpose(M))) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 color;

void main()
{
    // light and material properties
    vec3 light_pos1 = vec3(3,2,4);
    vec3 light_pos2 = vec3(-3,2,-4);
    vec3 light_color = vec3(1,1,1);
    vec3 material_color = color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient1 = 0.1*light_color;
    vec3 light_ambient2 = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // or can be material_color

    // ambient
    vec3 ambient = light_ambient1 * material_ambient;
    ambient += light_ambient2 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir1 = normalize(light_pos1 - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff1 = max(dot(normal, light_dir1), 0);
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse1 = diff1 * light_diffuse * material_diffuse;
    vec3 diffuse2 = diff2 * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir1 = reflect(-light_dir1, normal);
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec1 = pow( max(dot(view_dir, reflect_dir1), 0.0), material_shininess);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular1 = spec1 * light_specular * material_specular;
    vec3 specular2 = spec2 * light_specular * material_specular;

    vec3 color = ambient + diffuse1 + specular1 + diffuse2 + specular2;
    FragColor = vec4(color, 1.);
}
'''
class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform
    def set_shape_transform(self, shape_trans):
        self.shape_transform = shape_trans

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def draw_node(vao, node, VP, MVP_loc, color_loc):
    M = node.get_global_transform() * node.get_shape_transform()
    MVP = VP * M
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    if(key_one == True):
        glDrawArrays(GL_LINES, 0, 6)
    elif(key_two == True):
        glDrawArrays(GL_TRIANGLES, 0, 36)

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
    global use_ortho, cam_distance, motion_mode, key_one, key_two
    if action==GLFW_PRESS:
        if key==GLFW_KEY_V:
            if(use_ortho == True):
                use_ortho = False
                cam_distance = 7.0
            else:
                use_ortho = True
                cam_distance = 3.0
        elif key==GLFW_KEY_SPACE:
            motion_mode = True
        elif key==GLFW_KEY_1:
            key_one = True
            key_two = False
        elif key==GLFW_KEY_2:
            key_two = True
            key_one = False
            

def framebuffer_size_callback(window, width, height):
    global g_P
    glViewport(0,0,width, height)
    if use_ortho == True:
        ortho_height = height * .01
        ortho_width = ortho_height * width / height
        g_P = glm.ortho(-cam_distance * ortho_width * .1, cam_distance * ortho_width * .1, -cam_distance * ortho_height  * .1, cam_distance * ortho_height * .1, 
                        -10, 10)

def drop_callback(window, file):
    global channelList, Root, parent, offset, frameorder, Frames, FrameTime, allElements, Nodes_one, Nodes_two, vao_list_line, frame_list, frame_list_idx, motion_mode, is_scaled, vao_list_box

    f = open(file[0], 'r')
    lines = f.readlines()
    frame_list = []
    frameorder = []
    Root = 0
    parent = {}
    channelList = {}
    allElements = []
    Nodes_one = {}
    Nodes_two = {}
    frame_list_idx = 0
    stack = []
    vertices = []
    vao_list_line = []
    vao_list_box = []
    is_scaled = False

    isFrameTimeIn = False

    for line in lines:
        line = line.strip()
        line_list = line.split()
        if(len(line_list) < 1): continue
        if(line_list[0] == "ROOT"):
            Root = line_list[1]
            parent[Root] = Root
            frameorder.append(line_list[1])
            allElements.append(line_list[1])
            stack.append(line_list[1])
        elif(line_list[0] == "OFFSET"):
                weight = glm.vec3(1., 1., 1.)
                if(abs(float(line_list[1])) > 1 or abs(float(line_list[2])) > 1 or abs(float(line_list[3])) > 1):
                    weight = glm.vec3(.03, .03, .03)
                    is_scaled = True
                offset[stack[-1]] = glm.vec3(float(line_list[1]), float(line_list[2]), float(line_list[3])) * weight
        elif(line_list[0] == "JOINT"):
            parent[line_list[1]] = stack[-1]
            stack.append(line_list[1])
            frameorder.append(line_list[1])
            allElements.append(line_list[1])
        elif(line_list[0] == "}"): stack.pop()
        elif(line_list[0] == "CHANNELS"):
            temp = []
            for element in line_list:
                if(element != "CHANNELS"):
                    temp.append(element)
            channelList[stack[-1]] = temp
        elif(line_list[0] == "End"):
            parent[stack[-1]+"End"] = stack[-1]
            allElements.append(stack[-1]+"End")
            stack.append(stack[-1]+"End")
        elif(line_list[0] == "Frames:"): Frames = int(line_list[1])
        elif(line_list[0] == "Frame" and line_list[1] == "Time:"):
            FrameTime = float(line_list[2])
            isFrameTimeIn = True
        elif(isFrameTimeIn == True):
            frame_list.append(line)

    for element in allElements:
        if(element != Root):
            vertices = []
            pos = offset[element]
            vertices.append(0)
            vertices.append(0)
            vertices.append(0)
            vertices.append(1)
            vertices.append(1)
            vertices.append(1)
            vertices.append(pos.x)
            vertices.append(pos.y)
            vertices.append(pos.z)
            vertices.append(1)
            vertices.append(1)
            vertices.append(1)
            vao_list_line.append(pv.prepare_vao_obj(vertices))

    for element in allElements:
        if(element != Root):
            vertices = []
            th = 0
            ang = 0
            pos = offset[element]
            if(pos.x*pos.x + pos.z*pos.z != 0):
                th = np.arctan(pos.y / np.sqrt(pos.x*pos.x + pos.z*pos.z))
            if(pos.z != 0):
                ang = np.arctan(pos.x / pos.z)

            width = .05
            v3 = glm.vec3(- width*np.cos(ang) - width*np.sin(th)*np.sin(ang),
                         width*np.cos(th),
                         width*np.sin(ang) - width*np.sin(th)*np.cos(ang))
            v7 = glm.vec3(width*np.cos(ang) - width*np.sin(th)*np.sin(ang),
                          width*np.cos(th),
                          - width*np.sin(ang) - width*np.sin(th)*np.cos(ang))
            v2 = -v7
            v6 = -v3
            v0 = v3 + pos
            v4 = v7 + pos
            v5 = v6 + pos
            v1 = v2 + pos

            vec1 = v2 - v0
            vec2 = v1 - v0
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v0
            vertices.append(v0.x)
            vertices.append(v0.y)
            vertices.append(v0.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v2
            vertices.append(v2.x)
            vertices.append(v2.y)
            vertices.append(v2.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v1
            vertices.append(v1.x)
            vertices.append(v1.y)
            vertices.append(v1.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v3 - v0
            vec2 = v2 - v0
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v0
            vertices.append(v0.x)
            vertices.append(v0.y)
            vertices.append(v0.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v3
            vertices.append(v3.x)
            vertices.append(v3.y)
            vertices.append(v3.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v2
            vertices.append(v2.x)
            vertices.append(v2.y)
            vertices.append(v2.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            
            vec1 = v5 - v4
            vec2 = v6 - v4
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v4
            vertices.append(v4.x)
            vertices.append(v4.y)
            vertices.append(v4.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v5
            vertices.append(v5.x)
            vertices.append(v5.y)
            vertices.append(v5.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v6
            vertices.append(v6.x)
            vertices.append(v6.y)
            vertices.append(v6.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v6 - v4
            vec2 = v7 - v4
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v4
            vertices.append(v4.x)
            vertices.append(v4.y)
            vertices.append(v4.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v6
            vertices.append(v6.x)
            vertices.append(v6.y)
            vertices.append(v6.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v7
            vertices.append(v7.x)
            vertices.append(v7.y)
            vertices.append(v7.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v1 - v0
            vec2 = v5 - v0
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v0
            vertices.append(v0.x)
            vertices.append(v0.y)
            vertices.append(v0.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v1
            vertices.append(v1.x)
            vertices.append(v1.y)
            vertices.append(v1.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v5
            vertices.append(v5.x)
            vertices.append(v5.y)
            vertices.append(v5.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v5 - v0
            vec2 = v4 - v0
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v0
            vertices.append(v0.x)
            vertices.append(v0.y)
            vertices.append(v0.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v5
            vertices.append(v5.x)
            vertices.append(v5.y)
            vertices.append(v5.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v4
            vertices.append(v4.x)
            vertices.append(v4.y)
            vertices.append(v4.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v6 - v3
            vec2 = v2 - v3
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v3
            vertices.append(v3.x)
            vertices.append(v3.y)
            vertices.append(v3.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v6
            vertices.append(v6.x)
            vertices.append(v6.y)
            vertices.append(v6.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v2
            vertices.append(v2.x)
            vertices.append(v2.y)
            vertices.append(v2.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v7 - v3
            vec2 = v6 - v3
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v3
            vertices.append(v3.x)
            vertices.append(v3.y)
            vertices.append(v3.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v7
            vertices.append(v7.x)
            vertices.append(v7.y)
            vertices.append(v7.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v6
            vertices.append(v6.x)
            vertices.append(v6.y)
            vertices.append(v6.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v2 - v1
            vec2 = v6 - v1
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v1
            vertices.append(v1.x)
            vertices.append(v1.y)
            vertices.append(v1.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v2
            vertices.append(v2.x)
            vertices.append(v2.y)
            vertices.append(v2.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v6
            vertices.append(v6.x)
            vertices.append(v6.y)
            vertices.append(v6.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v6 - v1
            vec2 = v5 - v1
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v1
            vertices.append(v1.x)
            vertices.append(v1.y)
            vertices.append(v1.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v6
            vertices.append(v6.x)
            vertices.append(v6.y)
            vertices.append(v6.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v5
            vertices.append(v5.x)
            vertices.append(v5.y)
            vertices.append(v5.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v7 - v0
            vec2 = v3 - v0
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v0
            vertices.append(v0.x)
            vertices.append(v0.y)
            vertices.append(v0.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v7
            vertices.append(v7.x)
            vertices.append(v7.y)
            vertices.append(v7.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v3
            vertices.append(v3.x)
            vertices.append(v3.y)
            vertices.append(v3.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])

            vec1 = v4 - v0
            vec2 = v7 - v0
            cross_prod = glm.cross(vec1, vec2)
            vec_len = np.sqrt(cross_prod.x**2 + cross_prod.y**2 + cross_prod.z**2)
            unit = cross_prod
            if vec_len != 0: unit /= vec_len
            #v0
            vertices.append(v0.x)
            vertices.append(v0.y)
            vertices.append(v0.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v4
            vertices.append(v4.x)
            vertices.append(v4.y)
            vertices.append(v4.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            #v7
            vertices.append(v7.x)
            vertices.append(v7.y)
            vertices.append(v7.z)
            vertices.append(unit[0])
            vertices.append(unit[1])
            vertices.append(unit[2])
            vao_list_box.append(pv.prepare_vao_box(vertices))

    parent[Root] = None
    Nodes_one[Root] = Node(None, glm.translate(offset[Root]), glm.mat4(), glm.vec3(0,0,1))
    Nodes_two[Root] = Node(None, glm.translate(offset[Root]), glm.mat4(), glm.vec3(.31,.23,46))
    for element in frameorder:
        if(element != Root):
            Nodes_one[element] = Node(Nodes_one[parent[element]], glm.translate(offset[element]), glm.mat4(), glm.vec3(0,0,1))
            Nodes_two[element] = Node(Nodes_two[parent[element]], glm.translate(offset[element]), glm.mat4(), glm.vec3(.31,.23,46))
    frame_list_idx = 0
    motion_mode = False

    print("Filename:", file[0])
    print("Frames:", Frames)
    print("FPS:", FrameTime)
    print("Num of joints:", len(frameorder))
    print("List of joints:", frameorder)
    f.close()

def main():
    global g_P, Nodes, vao_list_line, allElements, frame_list_idx
    
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1000, 1000, '2019049716_안재국', None, None)

    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetDropCallback(window, drop_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    color_loc  = glGetUniformLocation(shader_program, 'color')
    
    # prepare vaos
    vao_frame = pv.prepare_vao_frame()
    vao_row_grid = pv.prepare_vao_row_grid()
    vao_col_grid = pv.prepare_vao_col_grid()
    vao_box = pv.prepare_vao_cube()

    # loop until the user closes the window
    g_P = glm.ortho(-cam_distance, cam_distance, -cam_distance, cam_distance, -10, 10)

    while not glfwWindowShouldClose(window):
        # render
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        if(use_ortho):
            P = g_P
        else: P = glm.perspective(45, 1000/1000, 1, 100)

        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        
        cam_pos=glm.vec3(
            cam_distance*np.cos(g_cam_height)*np.sin(g_cam_ang),
            cam_distance * np.sin(g_cam_height),
            cam_distance*np.cos(g_cam_height)*np.cos(g_cam_ang)
        )

        V_orbit = glm.lookAt(glm.vec3(flip_up_vector, 1, flip_up_vector) * cam_pos, glm.vec3(0,0,0), glm.vec3(0,1,0) * flip_up_vector)
        V_pan = glm.lookAt(glm.vec3(trans_x,trans_y, 0), glm.vec3(trans_x, trans_y, -1), glm.vec3(0,1,0))
        V = V_pan*V_orbit
        # current frame: P*V*I (now this is the world frame)

        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniform3f(color_loc, 1, 1, 1)

        M = glm.mat4()
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, cam_pos.x, cam_pos.y, cam_pos.z)

        t = glfwGetTime()
        if(Root != 0):
            if motion_mode == True:
                if t > FrameTime and frame_list_idx < int(Frames):
                    frame_idx = 0
                    now_frame = frame_list[frame_list_idx]
                    now_frame = now_frame.split()
                    weight = glm.vec3(.03, .03, .03)
                    for element in frameorder:
                        chan = channelList[element]
                        T = glm.mat4()
                        for i in range(1, int(chan[0]) + 1):
                            if(chan[i].upper() == "XPOSITION"):
                                if(is_scaled == True):
                                    T *= glm.translate((float(now_frame[frame_idx]), 0, 0) * weight)
                                else: T *= glm.translate((float(now_frame[frame_idx]), 0, 0))
                            elif(chan[i].upper() == "YPOSITION"):
                                if(is_scaled == True):
                                    T *= glm.translate((0,float(now_frame[frame_idx]), 0) * weight)
                                else: T *= glm.translate((0,float(now_frame[frame_idx]), 0))
                            elif(chan[i].upper() == "ZPOSITION"):
                                if(is_scaled == True):
                                    T *= glm.translate((0,0,float(now_frame[frame_idx]))* weight)
                                else: T *= glm.translate((0,0,float(now_frame[frame_idx])))
                            elif(chan[i].upper() == "XROTATION"):
                                T *= glm.rotate(np.radians(float(now_frame[frame_idx])), (1,0,0))
                            elif(chan[i].upper() == "YROTATION"):
                                T *= glm.rotate(np.radians(float(now_frame[frame_idx])), (0,1,0))
                            elif(chan[i].upper() == "ZROTATION"):
                                T *= glm.rotate(np.radians(float(now_frame[frame_idx])), (0,0,1))
                            frame_idx += 1
                        if(key_one == True):
                            Nodes_one[element].set_joint_transform(T)
                        elif(key_two == True):
                            Nodes_two[element].set_joint_transform(T)
                    glfwSetTime(0.0)
                    frame_list_idx += 1
                    if frame_list_idx == Frames - 1:
                        frame_list_idx = 0

            if(key_one == True):
                Nodes_one[Root].update_tree_global_transform()
            elif(key_two == True):
                Nodes_two[Root].update_tree_global_transform()
            for element in allElements:
                if(element == Root): continue
                if(key_one == True):
                    draw_node(vao_list_line[allElements.index(element) - 1], Nodes_one[parent[element]], P*V, MVP_loc, color_loc)
                elif(key_two == True):
                    draw_node(vao_list_box[allElements.index(element) - 1], Nodes_two[parent[element]], P*V, MVP_loc, color_loc)


        glUniform3f(color_loc, 1, 1, 1)
        for i in range (0,100):
            T = glm.translate(glm.vec3(0, 0, i*.5))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_row_grid)
            glDrawArrays(GL_LINES, 0, 6)

            T = glm.translate(glm.vec3(0, 0, -i*.5))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_row_grid)
            glDrawArrays(GL_LINES, 0, 6)
        
        for i in range (0, 100):
            T = glm.translate(glm.vec3(i*.5, 0, 0))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_col_grid)
            glDrawArrays(GL_LINES, 0, 6)

            T = glm.translate(glm.vec3(-i*.5, 0, 0))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_col_grid)
            glDrawArrays(GL_LINES, 0, 6)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()

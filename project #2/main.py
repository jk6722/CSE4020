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

use_ortho = False

vao_obj = 0
vao_count = 0

single_mesh_mode = False
wire_mode = False

g_P = 0

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
    def __init__(self, parent, scale, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.scale = scale
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform # if Root node

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_scale(self):
        return self.scale
    def get_color(self):
        return self.color

def draw_node(vao, node, VP, MVP_loc, color_loc, vtx_count):
    MVP = VP * node.get_global_transform() * glm.scale(node.get_scale())
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, vtx_count)

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
    global use_ortho, cam_distance, single_mesh_mode, wire_mode
    if action==GLFW_PRESS:
        if key==GLFW_KEY_V:
            if(use_ortho == True):
                use_ortho = False
                cam_distance = 7.0
            else:
                use_ortho = True
                cam_distance = 3.0
        elif key==GLFW_KEY_H:
            if single_mesh_mode == True:
                single_mesh_mode = False
            elif single_mesh_mode == False:
                single_mesh_mode = True
        elif key==GLFW_KEY_Z:
            if wire_mode == False:
                wire_mode = True
            else: wire_mode = False

def framebuffer_size_callback(window, width, height):
    global g_P
    glViewport(0,0,width, height)
    if use_ortho == True:
        ortho_height = height * .01
        ortho_width = ortho_height * width / height
        g_P = glm.ortho(-cam_distance * ortho_width * .1, cam_distance * ortho_width * .1, -cam_distance * ortho_height  * .1, cam_distance * ortho_height * .1, 
                        -10, 10)

def drop_callback(window, file):
    global vao_obj, vao_count, single_mesh_mode
    vao_count = 0
    face_count = 0
    face_with_three_vertices = 0
    face_with_four_vertices = 0
    face_with_over_vertices = 0

    f = open(file[0], 'r')
    print(file[0])
    lines = f.readlines()
    vertex_list = []
    normal_list = []
    vertices = []

    for line in lines:
        line = line.strip()
        string_list = line.split()
        if(len(string_list) < 1): continue

        if(string_list[0] == 'o'):
            print("Obj file name: ", line)
        elif(string_list[0] == 'v'):
            vertex_list.append(glm.vec3(float(string_list[1]), float(string_list[2]) ,float(string_list[3])))
        elif(string_list[0] == 'vn'):
            normal_list.append(glm.vec3(float(string_list[1]), float(string_list[2]), float(string_list[3])))
        elif(string_list[0] == 'f'):
            vertex_idx = []
            normal_idx = []
            face_count += 1
            isNormal =  False

            for i in range(1, len(string_list)):
                temp = string_list[i].split('//')
                if len(temp) == 1:
                    temp = string_list[i].split('/')
                vertex_idx.append(int(temp[0]) - 1)
                if len(temp) == 2:
                    normal_idx.append(int(temp[1]) - 1)
                    isNormal = True
                elif len(temp) == 3:
                    normal_idx.append(int(temp[2]) - 1)
                    isNormal = True

            vertex_num = len(vertex_idx)
            if vertex_num == 3:
                face_with_three_vertices += 1
            elif vertex_num == 4:
                face_with_four_vertices += 1
            elif vertex_num > 4:
                face_with_over_vertices +=1

            for i in range(0, vertex_num - 2):
                vao_count += 3
                for j in range(3):
                    vertices.append(vertex_list[vertex_idx[0]][j])
                if isNormal == True:
                    for j in range(3):
                        vertices.append(normal_list[normal_idx[0]][j])
                for j in range(3):
                    vertices.append(vertex_list[vertex_idx[1 + i]][j])
                if isNormal == True:
                    for j in range(3):
                        vertices.append(normal_list[normal_idx[1 + i]][j])
                for j in range(3): 
                    vertices.append(vertex_list[vertex_idx[2 + i]][j])
                if isNormal == True:
                    for j in range(3):
                        vertices.append(normal_list[normal_idx[2 + i]][j])

    print("Total number of faces: ", face_count)
    print("Number of faces with 3 vertices: ", face_with_three_vertices)
    print("Number of faces with 4 vertices: ", face_with_four_vertices)
    print("Number of faces with more than 4 vertices: ", face_with_over_vertices)
    vao_obj = pv.prepare_vao_obj(vertices)
    single_mesh_mode = True
    f.close()

def main():
    global target, g_P
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
    vao_road = pv.prepare_vao_hierarchy("road OBJ")
    vao_car = pv.prepare_vao_hierarchy("car OBJ")
    vao_wheel = pv.prepare_vao_hierarchy("wheel OBJ")
    vao_moon = pv.prepare_vao_hierarchy("moon OBJ")
    vao_star = pv.prepare_vao_hierarchy("star OBJ")
    vao_astronaut = pv.prepare_vao_hierarchy("astronaut OBJ")
    vao_heart = pv.prepare_vao_hierarchy("heart OBJ")

    road = Node(None, glm.vec3(1.,1.,3.), glm.vec3(1.,1.,1.))
    car = Node(road, glm.vec3(.5,.5,.5), glm.vec3(.0, .2, .5))
    wheel1 = Node(car, glm.vec3(.1,.1,.1), glm.vec3(.5,.5,.5))
    wheel2 = Node(car, glm.vec3(.1,.1,.1), glm.vec3(.5,.5,.5))
    wheel3 = Node(car, glm.vec3(.1,.1,.1), glm.vec3(.5,.5,.5))
    wheel4 = Node(car, glm.vec3(.1,.1,.1), glm.vec3(.5,.5,.5))
    moon = Node(road, glm.vec3(.1,.1,.1), glm.vec3(.8,.8,.8))
    star = Node(moon, glm.vec3(.1,.1,.1), glm.vec3(.5,.5,0))
    astronaut = Node(moon, glm.vec3(.01,.01,.01), glm.vec3(1,1,1))
    heart = Node(car, glm.vec3(.05,.05,.05), glm.vec3(1,0,0))
    
    go_forward = 1
    prev_sin = 0
    # loop until the user closes the window
    g_P = glm.ortho(-cam_distance, cam_distance, -cam_distance, cam_distance, -10, 10)
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        if wire_mode == True:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


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
        glUniform3f(color_loc, 1., 1., 1.)
        
        # draw current frame
        glBindVertexArray(vao_frame) #global frame
        glDrawArrays(GL_LINES, 0, 6)

        M = glm.mat4()
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, cam_pos.x, cam_pos.y, cam_pos.z)
        t = glfwGetTime()
        if single_mesh_mode == True:
            if(vao_obj != 0):
                glBindVertexArray(vao_obj)
                glDrawArrays(GL_TRIANGLES, 0, vao_count)
        elif single_mesh_mode == False:
            th = (np.sin(t) + 1)
            move = np.sin(t*.5)
            if(prev_sin < move):
                go_forward = 1
            else: go_forward = -1
            prev_sin = move
            rotate_road = np.sin(t*.3) * 25
            trans_car = move * 8
            trans_moon = np.sin(t*.3) * 8
            R = glm.rotate(go_forward * np.radians(t * 200), glm.vec3(1,0,0))
            road.set_transform(glm.rotate(np.radians(t * 10), glm.vec3(0,1,0)))
            car.set_transform(glm.translate(glm.vec3(-.4, .2, 0)) * glm.translate(glm.vec3(0,0,trans_car)))
            wheel1.set_transform(glm.translate(glm.vec3(1.4,.25, -1)) * R)
            wheel2.set_transform(glm.translate(glm.vec3(.4,.25,-1)) * R)
            wheel3.set_transform(glm.translate(glm.vec3(1.4,.25, -2.76)) * R)
            wheel4.set_transform(glm.translate(glm.vec3(.4,.25, -2.76)) * R)
            moon.set_transform(glm.translate(glm.vec3(0, 5, 0)) * glm.translate(glm.vec3(0,0,trans_moon)) * glm.rotate(np.radians(t*60), glm.vec3(0,1,0)) * glm.translate(glm.vec3(1,0,0)))
            star.set_transform(glm.rotate(np.radians(t*90), glm.vec3(1,1,0)) * glm.translate(glm.vec3(0,2,0)))
            astronaut.set_transform(glm.rotate(np.radians(t*90), glm.vec3(0,0,1)) * glm.translate(glm.vec3(0,1.5,0)) * glm.rotate(np.radians(t * 60), glm.vec3(0,1,0)))
            heart.set_transform(glm.translate(glm.vec3(.9,1,-2.5))*glm.translate(glm.vec3(0,th,0))*glm.rotate(-np.pi/2, glm.vec3(1,0,0)) * glm.rotate(np.radians(t*60), glm.vec3(0,0,1)))
            road.update_tree_global_transform()
            
            draw_node(vao_road[0], road, P*V, MVP_loc, color_loc, int(vao_road[1]))
            draw_node(vao_car[0], car, P*V, MVP_loc, color_loc, int(vao_car[1]))
            draw_node(vao_wheel[0], wheel1, P*V, MVP_loc, color_loc, int(vao_wheel[1]))
            draw_node(vao_wheel[0], wheel2, P*V, MVP_loc, color_loc, int(vao_wheel[1]))
            draw_node(vao_wheel[0], wheel3, P*V, MVP_loc, color_loc, int(vao_wheel[1]))
            draw_node(vao_wheel[0], wheel4, P*V, MVP_loc, color_loc, int(vao_wheel[1]))
            draw_node(vao_moon[0], moon, P*V, MVP_loc, color_loc, int(vao_moon[1]))
            draw_node(vao_star[0], star, P*V, MVP_loc, color_loc, int(vao_star[1]))
            draw_node(vao_astronaut[0], astronaut, P*V, MVP_loc, color_loc, int(vao_astronaut[1]))
            draw_node(vao_heart[0], heart, P*V, MVP_loc, color_loc, int(vao_heart[1]))

        glUniform3f(color_loc, 1, 1, 1)

        for i in range (0,50):
            T = glm.translate(glm.vec3(0, 0, i))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_row_grid)
            glDrawArrays(GL_LINES, 0, 6)

            T = glm.translate(glm.vec3(0, 0, -i))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_row_grid)
            glDrawArrays(GL_LINES, 0, 6)
        
        for i in range (0, 50):
            T = glm.translate(glm.vec3(i, 0, 0))
            M = T
            MVP = P*V*M
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
            glBindVertexArray(vao_col_grid)
            glDrawArrays(GL_LINES, 0, 6)

            T = glm.translate(glm.vec3(-i, 0, 0))
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

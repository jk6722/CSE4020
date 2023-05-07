from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         1.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 1.0, 0.0,  0.0, 1.0, 0.0, # y-axis end
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 1.0,  0.0, 0.0, 1.0, # z-axis end
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_row_grid():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -50.0, 0.0, 0.0,  .8, .8, .8, # x-axis start
         50.0, 0.0, 0.0,  .8, .8, .8, # x-axis end
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_col_grid():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, -50.0,  .8, .8, .8, # z-axis start
         0.0, 0.0, 50.0,   .8, .8, .8, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position            color
        -0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v0
         0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v2
         0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v1
                    
        -0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v0
        -0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v3
         0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v2
                    
        -0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v4
         0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v5
         0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v6
                    
        -0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v4
         0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v6
        -0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v7
                    
        -0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v0
         0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v1
         0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v5
                    
        -0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v0
         0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v5
        -0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v4
 
        -0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v3
         0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v6
         0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v2
                    
        -0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v3
        -0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v7
         0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v6
                    
         0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v1
         0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v2
         0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v6
                    
         0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v1
         0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v6
         0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v5
                    
        -0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v0
        -0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v7
        -0.5 , -0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v3
                    
        -0.5 ,  0.5 ,  0.5 ,  0.5, 0.5, 0.5, # v0
        -0.5 ,  0.5 , -0.5 ,  0.5, 0.5, 0.5, # v4
        -0.5 , -0.5 , -0.5 ,  0.5, 0.5, 0.5, # v7
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO
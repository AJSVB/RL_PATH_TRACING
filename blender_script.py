import bpy
import time
from random import randint



scene = bpy.context.scene
nodes = bpy.context.scene.node_tree.nodes

for frame in range(scene.frame_start, scene.frame_start + 1):
  for i in range(1000):
    bpy.context.scene.cycles.seed = i 
    bpy.context.scene.render.image_settings.color_mode ='RGB'
    bpy.context.scene.render.image_settings.file_format='PNG'
    bpy.context.scene.render.filepath = "~"  
    scene.render.filepath = '~/' + str(frame).zfill(4) + "-" + str(i).zfill(5) 
    scene.frame_set(frame)
    bpy.ops.render.render(write_still=True)
  


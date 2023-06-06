import sys
j= int(sys.argv[-1])
      
         
import bpy
import time
from random import randint


context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render
render.image_settings.color_mode = 'RGBA' 
render.film_transparent = True
layer = bpy.context.view_layer



nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links
render_layers = nodes.new('CompositorNodeRLayers')

scene = bpy.context.scene

temp = []

bpy.context.scene.render.filepath = "bubble/"  

render_file_path = "add"
for a in render_layers.outputs.keys():
            albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
            albedo_file_output.label = 'Albedo Output'
            albedo_file_output.base_path = ''
            albedo_file_output.file_slots[0].use_node_format = True
            albedo_file_output.format.file_format = 'PNG'
            albedo_file_output.format.color_mode = 'RGB'
            albedo_file_output.format.color_depth = '16'
            links.new(render_layers.outputs[a], albedo_file_output.inputs[0])
            albedo_file_output.file_slots[0].path = render_file_path +str(a).zfill(4)+ str(a)
            temp.append(albedo_file_output)


for frame in range(j, j + 1):
  scene.frame_set(frame)   
  bpy.ops.render.render(write_still=True)

import bpy
import time
from random import randint

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links
render_layers = nodes.new('CompositorNodeRLayers')
scene = bpy.context.scene
albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
albedo_file_output.base_path = "" 
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = 'PNG'
albedo_file_output.format.color_mode = 'RGB'
albedo_file_output.format.color_depth = '16'
links.new(render_layers.outputs["Image"], albedo_file_output.inputs[0])

for frame in range(scene.frame_start, scene.frame_start + 1):
  for i in range(920,1000):
    bpy.context.scene.cycles.seed = i 
    scene.frame_set(frame)
    albedo_file_output.file_slots[0].path = '~/' + str(frame).zfill(4) + "-" + str(i).zfill(5)+ ".png" 
    bpy.ops.render.render(write_still=True)
  


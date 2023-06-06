import sys
j= int(sys.argv[-1])
      
import bpy
import time
from random import randint
scene = bpy.context.scene
bpy.context.scene.render.image_settings.color_mode ='RGB'
bpy.context.scene.render.image_settings.file_format='PNG'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.compute_device = 'CUDA_0'
bpy.ops.wm.save_userpref()

bpy.context.scene.render.filepath = "bubble/"  
a = "bubble"

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


enable_gpus("CUDA")
bpy.context.scene.cycles.use_denoising = False
bpy.context.scene.cycles.use_preview_denoising=False
for frame in range(j, (j+1)):
  scene.frame_set(frame) 
  bpy.context.scene.cycles.device = 'GPU'
  bpy.context.scene.cycles.samples = 1000
  scene.render.filepath = a+'/gd' + str(frame).zfill(4)  
  bpy.ops.render.render(write_still=True)
  for i in range(8):
    bpy.context.scene.cycles.seed = i 
    bpy.context.scene.cycles.samples = 1 
    scene.render.filepath = a+'/'+str(i).zfill(2) +"-" + str(frame).zfill(4)  
    bpy.ops.render.render(write_still=True)
      
      
      
      

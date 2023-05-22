import sys
j= int(sys.argv[-1])
      
import bpy
import time
from random import randint

scene = bpy.context.scene

prefs = bpy.context.preferences.addons['cycles'].preferences
bpy.context.scene.render.use_compositing = True

prefs.compute_device_type = 'CUDA'
prefs.compute_device = 'CUDA_0'



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
for frame in range(j, (j+1)):
  scene.frame_set(frame) 
  bpy.context.scene.cycles.device = 'GPU'
  scene.render.filepath = "suntemple/"+str(frame).zfill(4)  
  bpy.ops.render.render(use_viewport=True)      
      
      
      

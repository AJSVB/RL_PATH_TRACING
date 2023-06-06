
import sys
i= int(sys.argv[-1])

import bpy
import time
from random import randint
scene = bpy.context.scene
scene.frame_set(i) 
bpy.context.scene.render.use_compositing = True
bpy.ops.render.render(use_viewport=True)

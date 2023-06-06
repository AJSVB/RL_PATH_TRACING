import os
len = 1201
#path = "/home/antoine/Downloads/EmeraldSquare_v4_1/untitled.blend" 
#path = "/home/antoine/Downloads/ZeroDay_v1/MEASURE_ONE/untitled.blend" #"~/datasets
path = "Downloads/SunTemple_v4/SunTemple/suntemple.blend" #"~/datasets/EmeraldSquare_v4_1/untitled.>
#path = "/home/antoine/Downloads/ripple_dreams_fields.blend"
path = "untitled.blend"
import sys
import torch
import numpy as np
import os
import minexr

def store_flow(path,out):    
#    file = OpenEXR.InputFile(path)
#    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
#    (R,G,B,A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B","A") ]

    with open(path, 'rb') as fp:
        reader = minexr.load(fp)
        temp = reader.select(['R','G','B','A'])
  

#    temp = torch.cat((f(R),f(G),f(B),f(A)),-1)
    print(np.array(temp).shape)
    temp = torch.Tensor(np.array(temp))[:720,:720]
    print(torch.min(temp))
    a=720
    b=720
    bb = torch.Tensor(np.tile(np.arange(b),(720,1)))
    aa = torch.Tensor(np.tile(np.arange(a).T,(720,1)).T)



    temp[:,:,0] = torch.Tensor((temp[:,:,0]*(2)+bb*2-b+1)/b)
    temp[:,:,1] = torch.Tensor((-temp[:,:,1]*(2) +aa*2-a+1)/a)

    flow =  (torch.Tensor(temp)[:,:,[0,1]].unsqueeze(0))
    torch.save(flow,out)
    print(path)
    os.system("rm "+path)

for i in range(0,len):
    os.system("blender "+ path +" --background --python generate_add.py -- " +str(i) )
    os.system("blender "+ path +" --background --python generate_gd.py -- " +str(i) )
    os.system("blender "+ path +" --background --python script0.py -- " +str(i) )
    name = str(i).zfill(4)
    store_flow("/tmp/Image"+name+".exr","/home/antoine/motions/"+name+".pt")


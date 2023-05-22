import os
len = 500
#path = "~/datasets/EmeraldSquare_v4_1/untitled2.blend"
#path = "~/datasets/SunTemple_v4/SunTemple/untitled.blend"
#path = "~/SunTemple_v4/SunTemple/untitled.blend"
#path = "~/datasets/ZeroDay_v1/MEASURE_ONE/untitled.blend"
path = "~/datasets/ripple_dreams_fields.blend"
for i in range(len):
#    os.system("./blender "+ path +" --background --python generate_gd.py -- " +str(i))
    os.system("./blender "+ path +" --background --python generate_add.py -- " +str(i))


import OpenEXR
import Imath
import array
import sys

def f(a):
    return torch.Tensor(a).reshape(720,1280).unsqueeze(-1)

def store_flow(path,out):    
    file = OpenEXR.InputFile(path)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B,A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B","A") ]
    temp = torch.cat((f(R),f(G),f(B),f(A)),-1)
    a=720
    b=1280
    bb = torch.Tensor(np.tile(np.arange(b),(720,1)))
    aa = torch.Tensor(np.tile(np.arange(a).T,(1280,1)).T)

    temp[:,:,0] = torch.Tensor((temp[:,:,0]*(2)+bb*2-b+1)/b)
    temp[:,:,1] = torch.Tensor((-temp[:,:,1]*(2) +aa*2-a+1)/a)

    flow =  (torch.Tensor(temp)[:,:,[0,1]].unsqueeze(0))

    torch.save(flow,out)









import OpenEXR
import Imath
import array
import sys
import torch
import numpy as np
def f(a):
    return torch.Tensor(a).reshape(720,1280).unsqueeze(-1)

def store_flow(path,out):    
    print(path)
    file = OpenEXR.InputFile(path)
    print(file.header()["channels"].keys())
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    ter= "ViewLayer.Combined."
    (R,G,B,A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in (ter+"R", ter+"G", ter+"B",ter+"A") ]
    temp = torch.cat((f(R),f(G),f(B),f(A)),-1)
    a=720
    b=1280
    bb = torch.Tensor(np.tile(np.arange(b),(720,1)))
    aa = torch.Tensor(np.tile(np.arange(a).T,(1280,1)).T)

    temp[:,:,0] = torch.Tensor((temp[:,:,0]*(2)+bb*2-b+1)/b)
    temp[:,:,1] = torch.Tensor((-temp[:,:,1]*(2) +aa*2-a+1)/a)

    flow =  (torch.Tensor(temp)[:,:,[0,1]].unsqueeze(0))

    torch.save(flow,out)

inf = lambda x : 'motions/'+str(x).zfill(4)+".exr"
o = lambda x : 'motions/'+str(x).zfill(4)+".pt"
for i in range(1,1400):
 store_flow(inf(i),o(i))


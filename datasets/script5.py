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
    os.system("rm /home/antoine/"+path)






import os
for i in range(410):
  name = str(i).zfill(4)
  store_flow("####/Image"+name+".exr","motions/"+name+".pt")

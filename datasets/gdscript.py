import sys
#j=int(sys.argv[1])
from PIL import Image
import numpy as np
import torch
path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
def f(i,transform):
    print("f"+str(i))
    img1 = transform(torch.Tensor(get_truth(path,i+1)))
    img2 = transform(torch.Tensor(get_truth(path,i)))
    import cv2
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_AFFINE
    termination_eps = 1e-6
    im1_gray = cv2.cvtColor(np.float32(img1),cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(np.float32(img2),cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000,  termination_eps)
    try:
     (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)
    except cv2.error:
     warp_matrix = np.ones(warp_matrix.shape) #cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-5))

#        print("error, keep same warl_matrix")
   
    warp_matrix[:,2] = warp_matrix[:,2]/((2*720,2*720)) 
    return warp_matrix

def get_truth(path,frame_number):
    image= Image.open(path + "gd"+str(frame_number).zfill(4)+".png")
    #x = TF.to_tensor(image)
    return np.array(image)[:720,:720]/255.

a=[f(i) for i in range(j*100+1,(j+1)*100+1)]
a=np.array(a)
np.save("temp"+str(j)+".npy",a)
print(j)

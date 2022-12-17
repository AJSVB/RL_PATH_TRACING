import sys
from PIL import Image
import numpy as np
import torch
path = "/home/ascardigli/blender-3.2.2-linux-x64/suntemple/"
def f(i):
    print("f"+str(i))
    img1 = torch.Tensor(get_truth(path,i))
    img2 = torch.Tensor(get_truth(path,i+1))
    import cv2
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_mode = 2 #cv2.MOTION_TRANSLATION #MOTION_AFFINE
    termination_eps = 1e-10
    if i==306 or i ==988 or i == 1065:
      termination_eps=1
 
    im1_gray = cv2.cvtColor(np.float32(img1),cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(np.float32(img2),cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100,  termination_eps)
    try:
     (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)
    except cv2.error:
     print("on va pas la")
     warp_matrix = 10*np.ones(warp_matrix.shape) #cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-5))
    warp_matrix[:,2] = warp_matrix[:,2]/((2*720,2*720)) 
    return warp_matrix

def get_truth(path,frame_number):
    image= Image.open(path + "gd"+str(frame_number).zfill(4)+".png")
    return np.array(image)[:720,:720]/255.
def get_add(a,b,c):
    image= Image.open(a+b+c)
    z= np.array(image)[:720,:720]
    b=np.min(z)
    a=np.max(z)
    if b==a:
      a = a if a else 1.
      return z/a
    else:
      return (z-b)/(a-b)*1.

def get_aux(path,frame_number):
    f=path + "add"
    end = str(frame_number).zfill(4)+".png"
    imgs = get_add(f,2*"Denoising Albedo",end)
    return imgs

import time
a=[f(i) for i in range(0,1200)]
a=np.array(a)
np.save("temp1200.npy",a)

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from filelock import FileLock
from gym import Env, spaces
from PIL import Image
from pytorch_msssim import MS_SSIM, SSIM, ms_ssim, ssim
from torch.utils.data import Dataset
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import mean_squared_error
from torchvision.transforms import functional as F

from physicalsimulation import PhysicSimulation
from training import train

psnr = PeakSignalNoiseRatio().cuda(0)

class Spec:
   def __init__(self,max_episode_steps):
    self.max_episode_steps = max_episode_steps
    self.id = "3Drenderingenv"

class CustomEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config):
    super(CustomEnv, self).__init__()
    self.spp = env_config['spp']
    self.sppps = env_config["sppps"]
    self.mode = env_config["mode"]
    self.HEIGHT = 720 
    self.WIDTH =   720
    self.max = 20

    out_space=41
    if self.mode == "ntas":
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(12,3)
     out_space=12
    elif "notp" in self.mode:
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(33,0)
     out_space=9
    elif self.mode == "dasr":
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker(12,0)
     out_space=9
    else:
     self.model,self.data,self.criterion,self.optimizer,self.scheduler = train.main_worker()

    self.model=self.model.to("cuda:0")
    self.criterion = self.criterion.to("cuda:0")
    self.offset=0

    self.simulation = PhysicSimulation(self)
    self.action_space = spaces.Box(low=0,high=1,shape=(int(self.HEIGHT*self.WIDTH),))
    self.observation_space = spaces.Box(low=-1.0001, high=1.0001, shape=
                    (out_space,int(self.HEIGHT),int(self.WIDTH)), dtype=np.float32) #MACHINE PRECISION

    self.spec = Spec(self.max)
    self.top=0
    self.a=time.time()
    self.mses = []
    self.psnrs = []
    self.sum=0
    self.cntr=0
    with open("comp/"+str(self.spp)+'mses'+self.mode+'.txt', 'w') as fp:
        fp.write("\n")
    with open("comp/"+str(self.spp)+'psnrs'+self.mode+'.txt', 'w') as fp:
        fp.write("\n")

  def time(self):
    self.sum+=time.time() - self.a
    self.a= time.time()
    self.cntr+=1
    if self.cntr ==100:
      print(self.sum/100)
      self.cntr=0
      self.sum=0   

  def step(self, action):
    self.time()

    self.simulation.new(self.simulation.count)
    self.simulation.simulate(action)
    new = self.simulation.out(self.simulation.render())
    observation,gd = self.simulation.observe()
    loss= self.simulation.loss
    new1=1-loss

    if  self.bool:
     te = str((self.simulation.offset%100)+self.simulation.count)
     save(new.cpu(),"images/"+str(self.spp)+"new"+te+"_"+str(self.mode)+".png")  
     self.mses.append(mean_squared_error(new,gd).cpu())
     self.psnrs.append(psnr(new,gd).cpu())

    reward = 10**(new1)
    done = self.spec.max_episode_steps <= self.simulation.count
    return observation.numpy(),reward.detach().numpy(),done,{}
 
  def reset(self):
    self.bool=False
    if self.offset%self.simulation.number >= 800 and ((self.offset%self.simulation.number)//20*20)<900 : #was between 800 and 900
      self.bool=True
    self.simulation = PhysicSimulation(self)
    temp ,_= self.simulation.observe()

    if self.bool:
     with open("comp/"+str(self.spp)+'mses'+self.mode+'.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.mses))
         fp.write("\n")
     with open("comp/"+str(self.spp)+'psnrs'+self.mode+'.txt', 'a') as fp:
         fp.write("\n".join(str(item.item()) for item in self.psnrs))
         fp.write("\n")
    self.mses=[]
    self.psnrs=[]
    self.offset+=21
    if self.offset%20==4:
      self.offset-=64
    print(self.offset)
    return temp.numpy()
    
def save(data,name):
    img= T.ToPILImage()(data)
    if img.mode != 'RGB':
     img = img.convert('RGB')
    img.save(name)
	

import unet3
import torchvision.transforms as T
import random
import torch
import numpy as np
from torch.autograd import Function
from torch.autograd import Variable
import random
import os
def nostate(st):
    """checks if the mode is stateless or not
    
    Args:
        st (str): the mode

    Returns:
        bool: whether or not the mode is stateless
    """    
    return "notp" in st or "dasr" == st

def save(data, name):
    """helper function that saves a picture given a tensor

    Args:
        data (tensor): tensor of the image
        name (str): name of the saved picture
    """    
    data[data < 0] == torch.max(data)
    img = T.ToPILImage()(data)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(name)


"""
The rendering intergrator for gradient backpropagation support.
"""


class Render(Function):
    """The approximated gradient as described in Deep adaptive sampling for low count rendering
    """    
    # 
    @staticmethod
    def forward(ctx, x, sim):
        """forward pass

        Args:
            ctx (tensor): internal default argument
            x (tensor): the sampling importance recommendation
            sim (simulation object): reference to the whole simulation object

        Returns:
            tensor: output of the forward pass
        """        
        x = torch.flatten(x)
        M = sim.spp * sim.WIDTH * sim.HEIGHT
        e = x - torch.min(x)  # torch.exp(x)
        s = torch.sum(e)
        s = torch.floor((M * e) / s + 0.5).clamp(min=1)
        s[s < 0] = -1
        s[s > 8] = 8
        sim.s = s
        if random.random() < 0.01:
            print(torch.mean(s))
            print(torch.var(s))
        observations = sim.data.generate(sim.dataset, s.type(torch.long), sim.count)
        obs = observations.reshape(8, -1)
        mask = (
            obs != -1
        )  # in our implementation, -1 is the value for no sampling, we therefore need to mask those values out
        obs = (obs * mask).sum(dim=0) / mask.sum(dim=0)
        obs = obs.reshape(3, 720, 720)
        ctx.save_for_backward((sim.gd - obs) / s.reshape(1, 720, 720).expand(3, -1, -1))
        # ours with grad outputs the observations (3*8 channels), whereas ntas and dasr output the aggregated image (8 channels)
        if sim.mode == "grad":
            return observations
        else:
            return obs


    @staticmethod
    def backward(ctx, dL_dout):
        """Static method that we do not call in our code, but that is used internally by pytorch for the backward pass

        Args:
            ctx (_type_): initial argument
            dL_dout (_type_): initial argument

        Returns:
            _type_: approximated gradient as precomputed in the forward pass, according to DASR paper
        """    
        dS_dn = ctx.saved_tensors[0]
        if len(dL_dout.shape) == 3:
            dL_dout = dL_dout.unsqueeze(0)
        dL_din = torch.sum(torch.mul(dL_dout, Variable(dS_dn)), dim=1, keepdim=True)
        return tuple([dL_din, None])




class PhysicSimulation:
    def __init__(self, sel):
        """This class makes the link between the dataset, and our python framwork.

        Args:
            sel (Env): the custom environment
        """        
        self.number = 480 #3480  # total number of frames
        self.spp,  self.HEIGHT, self.WIDTH, self.offset, self.mode = (
            sel.spp,
            sel.HEIGHT,
            sel.WIDTH,
            int((sel.offset // 20) * 20) % self.number,
            sel.mode,
        )
        self.model, self.data, self.criterion, self.optimizer, self.scheduler = (
            sel.model,
            sel.data,
            sel.criterion,
            sel.optimizer,
            sel.scheduler,
        )
        self.interval = sel.interval
        self.reset()
        self.new(-1)
        self.shape = [8, self.HEIGHT, self.WIDTH]
        if self.mode == "imcduni":
            self.fcn = unet3.FCN().cuda()


    def reset(self):
        """Called during the initialisation of the class, but also after/before every animation of 20 equential frames.
        Resets the observations and state. 
        """        
        self.observations = -1 * torch.ones([8, 3, self.HEIGHT, self.WIDTH])
        self.updated = True
        self.denoised = torch.zeros([1, 3, self.HEIGHT, self.WIDTH]).cuda(0)
        self.count = -1
        self.loss = 0
        self.s = None
        if self.mode == "ntas":
            self.state = -1 * torch.ones([3, self.HEIGHT, self.WIDTH]).cuda(0)
        else:
            self.state = -1 * torch.ones([32, self.HEIGHT, self.WIDTH]).cuda(0)
        if self.mode == "imcduni":
            self.state = -1 * torch.ones([3, self.HEIGHT, self.WIDTH]).cuda(0)
        lis = []
        self.perm = lambda x: x
        if random.random() > 0.5:  # image augmentation
            lis.append(T.functional.hflip)
        if random.random() > 0.5:
            lis.append(T.functional.vflip)
        if self.inval():  # no augmentation in validation mode
            lis = []
        self.transform = T.Compose(lis)

    def new(self, i):
        """Is called to take from the dataset the precomputed raytraced images and related additional data/ground truth. 

        Args:
            i (int): the index of the frame
        """        
        if i == -1:
            self.nextadd, self.gd = self.data.get(i + 1 + self.offset)
            self.nextadd = self.transform(
                torch.Tensor(self.nextadd).permute(2, 0, 1).cuda(0)
            )
            self.gd = self.transform(torch.Tensor(self.gd).permute(2, 0, 1).cuda(0))
        else:
            if self.mode == "ntas":
                self.olddenoised = self.denoised
                self.oldgd = self.gd

            self.dataset = self.transform(self.data.data(i + self.offset))
            self.add, self.gd = self.data.get(i + self.offset)
            self.add = self.transform(torch.Tensor(self.add).permute(2, 0, 1).cuda(0))
            self.gd = self.transform(torch.Tensor(self.gd).permute(2, 0, 1).cuda(0))
            self.nextadd, _ = self.data.get(i + 1 + self.offset)
            self.nextadd = self.transform(
                torch.Tensor(self.nextadd).permute(2, 0, 1).cuda(0)
            )

    def round_retain_sum(self, x, N):
        """Rounds a list of real numbers into integers such that the total sum is preserved 
        by finding the correct threshold value for rounding up or down.

        Args:
            x (numpy array): list of floats
            N (_type_): The sum we wish our output list to have 

        Returns:
            numpy array: the rounded list
        """        
        N = np.round(N).astype(int)
        y = x.type(torch.int)
        M = torch.sum(y)
        K = N - M
        z = x - y
        if K != 0:
            idx = torch.topk(z, K, sorted=False).indices
            y[idx] += 1
        return y

    def simulate(self, x):
        """extracts rendered information given the precomputed ray traced images and the recommendation from networks. 

        Args:
            x (numpy/tensor): the sampling recommendations
        """        
        if "grad" in self.mode or "dasr" in self.mode or "ntas" in self.mode:
            self.observations = Render.apply(x, self)
        else:
            if not "uni" in self.mode:
                x = torch.Tensor(x).cuda(0)
                x = x - torch.min(x)
                x = torch.flatten(x).type(torch.float64)
                N = torch.sum(x)
                temp = self.spp * self.WIDTH * self.HEIGHT
                x = x * temp / N
                N = temp
                s = self.round_retain_sum(x, N)
            else:
                s = x
            s[s < 0] = -1
            s[s > 8] = 8
            self.s = s
            self.observations = self.data.generate(self.dataset, s, self.count)
        self.updated = False

    def inval(self, offset=None):
        """returns true if current animation is part of the training or testing set

        Args:
            offset (int, optional): if not none, overwrite the offset for the calculation of the interval. Defaults to None

        Returns:
            boolean: whether or not we are in the validation set
        """        
        if offset is None:
            offset = self.offset
        return offset >= self.interval[0] and offset < self.interval[1]
        
    def render(self):
        """Outputs the denoised output image given the observed ray traced images. 
            This method calls the forward method of the denoiser if a denoiser is used,
            backpropagates the loss to the denoiser, and stores the loss so that it can
            be used by the RL framework

        Returns:
            tensor: the rendered image
        """        
        if not self.updated:
            self.optimizer.zero_grad()
            if self.mode == "D":
                self.observations = torch.mean(self.observations, 0)  
            self.observations = self.observations.reshape(-1, *self.shape[-2:])
            if self.mode == "imcduni":
                self.state, _ = self.fcn(
                    torch.cat((self.state, self.add, self.observations)), None, None
                )
            m1 = self.observations
            m2 = self.add
            m3 = self.state
            if nostate(self.mode) or self.mode == "notp1":
                input = torch.cat((m1, m2), 0).unsqueeze(0)
            elif self.mode == "imcduni":
                self.state1 = self.state1.detach()
                input = torch.cat((self.state, self.state1), 0).unsqueeze(0)
            else:
                input = torch.cat((m1, m2, m3), 0).unsqueeze(0)
            if self.mode == "imcduni":
                self.denoised, _ = self.model(input)
                self.state1 = self.denoised
            else:
                self.denoised, self.state = self.model(input)
            if self.mode == "ntas":
                self.state = self.denoised
            if self.mode in ["ntas", "dasr"]:
                self.criterion = torch.nn.L1Loss()
            loss = self.criterion(self.denoised, self.gd.unsqueeze(0))
            if self.mode == "ntas":  # ntas has a temporal component in the loss
                loss += self.criterion(
                    self.denoised - self.olddenoised,
                    self.gd.unsqueeze(0) - self.oldgd.unsqueeze(0),
                )
            if not self.inval():
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            self.denoised = torch.clip(self.denoised.detach(), 0, 1)
            self.state = torch.clip(self.state.detach(), -1, 1)
            if self.mode == "ntas":
                self.olddenoised = self.denoised
        self.updated = True
        self.loss = loss.detach().cpu()

        if self.offset>100 and self.offset<200:
           t = str(self.offset+self.count-1)
           print(t)
           temp = self.denoised[0].to(torch.float).detach().cpu()
           os.system("rm images/"+t+self.mode+str(self.spp)+"out.png")
           save(temp,"images/"+t+self.mode+str(self.spp)+"out.png")
        return self.denoised

    def save(data, name):
     data[data < 0] == torch.max(data)
     img = T.ToPILImage()(data)#(data/torch.max(data)*255)
     if img.mode != 'RGB':
      img = img.convert('RGB')
     img.save(name)


    def observe(self):
        """outputs the observation (state) of the RL agent as well as the ground truth image

        Returns:
            tensor: the observation from the environment and the ground truth image
        """        
        self.state = self.state.to(torch.float)
        if self.count > -1:
            # we need to cancel the transformation for using the motion vectors correctly
            self.state = self.transform(self.state)
            self.state = self.data.translation(
                self.count + self.offset, self.state, self.perm
            )
            self.state = self.transform(self.state)
        m2 = self.nextadd
        m3 = self.state.detach()
        if nostate(self.mode):
            input = m2
        elif "imcduni" == self.mode:
            self.state1 = self.transform(self.state1)
            self.state1 = self.data.translation(
                self.count + self.offset, self.state1, self.perm
            )
            self.state1 = self.transform(self.state1)
            input = m2
        elif "notp1" == self.mode:
            obs = self.transform(self.observations.cuda(0))
            obs = self.data.translation(self.count + self.offset, obs, self.perm)
            obs = self.transform(obs)
            input = torch.cat((m2, obs), 0)
        else:
            input = torch.cat((m2, m3), 0)
        self.count += 1
        return input.cpu(), self.gd

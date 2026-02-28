import torch.nn as nn
import torch
from torch.nn import functional as F
from utils.Geometry import LineSegement,LocalAxis,Geometry1D
import numpy as np



class Embedding:
    def __init__(self,
                 HeavisideZero = 0):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.set_Heaviside(HeavisideZero)

    def getGamma(self,xy)->torch.Tensor:... 

    def set_no_grad(self):

        self.getGamma_grad = self.getGamma

        def getGamma_detach(xy):
            return self.getGamma_grad(xy).detach() 
        self.getGamma = getGamma_detach

    def set_Heaviside(self,HeavisideZero):

        def HeavisideZero0(x):
            return torch.sign(torch.relu(x))

        def HeavisideZero1(x):
            return 1 - torch.sign(torch.relu(-x))
        
        Heavisides = [HeavisideZero0,HeavisideZero1]
        self.Heaviside = Heavisides[HeavisideZero]

    
    def sign(self,x):
        return (self.Heaviside(x) - 0.5 ) * 2
    
    def zero(self,x,y):
        return torch.zeros_like(x)

    def one(self,x,y):
        return torch.ones_like(x)
    
    def neg_one(self,x,y):
        return - self.one(x,y)

class multiEmbedding(Embedding):
    '''拼接多个Embedding'''
    def __init__(self,GammaList:list[Embedding]):
        super().__init__()
        self.GammaList = GammaList

    def getGamma(self,xy):
        Gamma = torch.cat(list(map(lambda x:x.getGamma(xy) ,self.GammaList)),dim=1)
        return Gamma

class extendAxisNet(nn.Module):
    def __init__(self,net:nn.Module,
                 extendAxis:Embedding) -> None:
        super().__init__()
        self.net = net
        self.extendAxis = extendAxis
    
    def forward(self,xy):
        Gamma = self.extendAxis.getGamma(xy)
        axis = torch.cat((xy,Gamma),dim = 1)
        return self.net(axis)    

    def infer(self,axis):
        return self.net(axis)    
    
    def set_extend_axis(self,extendAxis:Embedding):
        self.extendAxis = extendAxis


class extendOutputNet(nn.Module):
    '''把Embedding放到输出'''
    def __init__(self,u_net:nn.Module,v_net:nn.Module,
                 EmbeddingGamma:Embedding):
        super().__init__()
        self.u_net = u_net
        self.v_net = v_net
        self.EmbeddingGamma = EmbeddingGamma

    def forward(self,xy):
        Gamma = self.EmbeddingGamma.getGamma(xy)
        u_Embedding = torch.sum(self.u_net(xy) * Gamma,dim=1)
        v_Embedding = torch.sum(self.v_net(xy) * Gamma,dim=1)
        return [u_Embedding,v_Embedding]

class InnerSurface(Embedding):

    def __init__(self, local_axis: LocalAxis, a , levelset):
        '''不含裂尖的内部裂纹面'''
        super().__init__()
        self.local_axis = local_axis
        self.a = torch.tensor(a)
        self.levelset = levelset
        self.set_Heaviside(1)
    
    def psi(self,local_x):
        # 用于判断点是否在裂尖区域内，返回1或0
        return self.Heaviside(self.a - torch.abs(local_x))
    
    def H(self,local_y):
        return self.sign(local_y)

    def getGamma(self,xy):
        x = xy[...,0]; y = xy[...,1]

        H = self.H(-self.levelset(x,y))
        local_x , local_y = self.local_axis.cartesianToLocal(x,y)
        psi = self.psi(local_x)

        gamma = H * psi
        return gamma.unsqueeze(-1) 
    

class multiInnerSurfaces(Embedding):

    def __init__(self,points:list[list]):
        '''多段不含裂尖的内部裂纹面'''    
        self.surfaces = []
        for i in range(len(points)-1):
            x0 = points[i][0] ; y0 = points[i][1]
            x1 = points[i+1][0] ; y1 = points[i+1][1]

            self.surfaces.append(InnerSurface(
                local_axis=LocalAxis(x0 = (x0 + x1)/2,
                                        y0 = (y0 + y1)/2,
                                        beta=3.14159/2),
                a = abs(y1 - y0)/2,
                levelset=LineSegement([x0,y0],[x1,y1]).levelset))
            
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.a = torch.sum(torch.stack([self.surfaces[i].a 
                                        for i in range(len(self.surfaces))]).float().to(device=self.device))
        
        self.set_Heaviside(1)
        
    def set_Heaviside(self, HeavisideZero):
        for surface in self.surfaces:
            surface.set_Heaviside(HeavisideZero)
        super().set_Heaviside(HeavisideZero)

    def getGamma(self,xy):
        # torch.sign用来过滤两条边上的重合点处导致的重复计算
        return torch.sign(torch.sum(torch.cat(list(map(lambda x:x.getGamma(xy) ,self.surfaces)),dim=1),dim=1))

class CrackEmbedding(Embedding):
    def __init__(self,  levelset,
                        xy0, xy1,
                        tip,
                        left_beta,
                        right_beta):
        super().__init__()
        self.levelset = levelset

        self.left_beta = left_beta
        self.right_beta = right_beta
        self.left_psi = LineSegement.init_theta(xy0,left_beta-np.pi/2).levelset_dist
        self.right_psi =  LineSegement.init_theta(xy1,right_beta+np.pi/2).levelset_dist 
        self.a = LineSegement(xy0,xy1).length
        self.norm = self.a **2
        if tip == 'left':
            self.right_psi = self.one
        elif tip == 'right':
            self.left_psi = self.one
        elif tip == 'both':
            self.norm = (self.norm/4)**2
        else:
            raise Exception() 
        
        self.getH = self.getH_standard
    
    def getPSI(self,xy):
        x = xy[...,0]; y = xy[...,1]
        return F.relu(- self.left_psi(x,y) * self.right_psi(x,y)) **2  / self.norm
    
    def getH_standard(self,xy):
        x = xy[...,0]; y = xy[...,1]
        ls = self.levelset(x,y)
        return self.get_crack_sign(ls)
    
    def get_crack_sign(self,ls):
        return self.sign(-ls)
    
    def getGamma(self, xy):
        H = self.getH(xy) 
        PSI = self.getPSI(xy)
        return (H * PSI).unsqueeze(-1)
    
    def set_ls(self,ls:float):
        '''便于求应力强度因子设置ls为定值'''
        def constant(xy):
            x = xy[...,0]; y = xy[...,1]
            return self.one(x,y) * ls

        self.getH = constant

    def restore_ls(self):
        self.getH = self.getH_standard
        
    
class LineCrackEmbedding(CrackEmbedding):
    def __init__(self, xy0, xy1, tip):
        self.Line = LineSegement(xy0,xy1)
        super().__init__(levelset = self.Line.levelset_dist,
                         xy0=xy0, xy1=xy1, tip=tip,
                         left_beta = self.Line.tangent_theta + np.pi,
                         right_beta = self.Line.tangent_theta)
        

class multiLineCrackEmbedding(Embedding):
    def __init__(self,points:list[list],tip='both'):
        '''仅在最后一段折线段考虑裂尖,在内部裂纹面上表面为1,下表面为-1'''
        
        self.xy0 = points[0]
        self.xy1 = points[-1]

        self.MAX_X = 100.0
        if self.xy0[-1] > self.MAX_X: raise Exception('Please change self.MAX_X')

        if tip == 'right':

            inner_points = points+[[-5.0, 100.0]] #偷懒，直接把域最右边的点视为裂纹面的最后一个点，避免斜裂尖的一些处理麻烦
            self.inner_surfaces = multiInnerSurfaces(inner_points)

            self.right_tip = LineCrackEmbedding(points[-2],points[-1],tip='right')        

            self.split_line = LineSegement.init_theta(points[-2],self.right_tip.Line.tangent_theta+np.pi/2)

        else:
            raise Exception('Not Implemented!')


        self.set_Heaviside(0)

    def getGamma(self, xy) -> torch.Tensor:
        '''判断是否位于裂尖部分'''
        left_H = self.Heaviside(self.split_line.levelset(xy[...,0],xy[...,1])).unsqueeze(-1)
        inner_basis = self.inner_surfaces.getGamma(xy).unsqueeze(-1) * (1-left_H)
        right_tip = self.right_tip.getGamma(xy) * left_H
        return inner_basis + right_tip

    
    def set_ls(self,ls:float):
        '''便于求应力强度因子设置ls为定值'''
        self.right_tip.set_ls(ls)

    def restore_ls(self):
        self.right_tip.restore_ls()


class InterfaceEmbedding(Embedding):
    def __init__(self,geometry:Geometry1D):
        super().__init__()
        self.levelset = geometry.levelset

    def getGamma(self,xy):
        x = xy[...,0]; y = xy[...,1]
        ls = self.levelset(x,y)
        ls = torch.where(ls>0,ls,-ls)
        return ls.unsqueeze(-1)
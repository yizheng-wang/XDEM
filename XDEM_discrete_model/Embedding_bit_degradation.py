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
            return torch.sign(torch.relu(x)) # torch.sign是-1或者1，如果是0输出就为0

        def HeavisideZero1(x):
            return 1 - torch.sign(torch.relu(-x))
        
        Heavisides = [HeavisideZero0,HeavisideZero1]
        self.Heaviside = Heavisides[HeavisideZero]

    
    def sign(self,x):
        return (self.Heaviside(x) - 0.5 ) * 2 # 变成-1和1，heaviside是0或者1，小于等于0的输出为0
    
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
        # 试一试把不同的Gamma相加
        # Gamma = torch.sum(Gamma,dim=1).unsqueeze(-1)
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
    def __init__(self, local_axis: LocalAxis, a, levelset,
                 decay_alpha: float = 10.0,   # 衰减强度
                 decay_p: float = 1.0,        # 幂指数(1=指数, 2=高斯)
                 decay_ell: float | None = None):  # 特征长度, None 表示不用归一化
        super().__init__()
        self.local_axis = local_axis
        self.a = torch.tensor(a)
        self.levelset = levelset                   # 要传有符号“距离”函数
        self.decay_alpha = float(decay_alpha)
        self.decay_p = float(decay_p)
        self.decay_ell = float(decay_ell) if decay_ell is not None else None
        self.set_Heaviside(1)

    def psi(self, local_x):
        # 只在当前线段投影范围内生效
        return self.Heaviside(self.a - torch.abs(local_x))

    def getGamma(self, xy):
        x = xy[..., 0]; y = xy[..., 1]
        # ls 是到“线段”的有符号距离；上/下表面用其符号确定
        ls = self.levelset(x, y)
        H  = self.sign(-ls)                         # 上表面 +1，下表面 -1
        local_x, _ = self.local_axis.cartesianToLocal(x, y)
        win = self.psi(local_x)                     # 仅线段投影范围内

        # 距离与衰减
        dist = torch.abs(ls)
        if self.decay_ell is None:
            decay = torch.exp(- self.decay_alpha * (dist ** self.decay_p))
        else:
            decay = torch.exp(- self.decay_alpha * ((dist / self.decay_ell) ** self.decay_p))

        gamma = H * win * decay
        return gamma.unsqueeze(-1)
    

class multiInnerSurfaces(Embedding):
    def __init__(self, points: list[list],
                 decay_alpha: float = 10.0,
                 decay_p: float = 1.0,
                 decay_ell: float | None = None):
        self.surfaces = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            seg = LineSegement([x0, y0], [x1, y1])
            self.surfaces.append(
                InnerSurface(
                    local_axis=LocalAxis(x0=(x0 + x1) / 2, y0=(y0 + y1) / 2, beta=3.14159/2),
                    a = abs(y1 - y0)/2,
                    levelset=seg.levelset_dist,    # ← 关键：用线段“有符号距离”
                    decay_alpha=decay_alpha,
                    decay_p=decay_p,
                    decay_ell=decay_ell
                )
            )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.a = torch.sum(torch.stack([s.a for s in self.surfaces]).float().to(self.device))
        self.set_Heaviside(1)

    def set_Heaviside(self, HeavisideZero):
        for s in self.surfaces:
            s.set_Heaviside(HeavisideZero)
        super().set_Heaviside(HeavisideZero)

    def getGamma(self, xy):
        gammas = torch.cat([s.getGamma(xy) for s in self.surfaces], dim=1)
        gsum = torch.sum(gammas, dim=1, keepdim=True)
        # 压缩避免拼接处 >1 或 <-1；也可用 clamp(-1,1)
        # return gsum
        return torch.tanh(gsum)   # 或者：return torch.clamp(gsum, -1.0, 1.0)

class CrackEmbedding(Embedding):
    def __init__(self,  levelset,
                        xy0, xy1,
                        tip,
                        left_beta,
                        right_beta,
                        decay_alpha: float = 10.0):
        super().__init__()
        self.levelset = levelset # 这实际上是距离函数
        
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
        self.decay_alpha = decay_alpha
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
        distance = torch.abs(self.levelset(xy[...,0],xy[...,1])) # 计算和裂纹延长线的最短距离 
        return (H * PSI * torch.exp(-self.decay_alpha*distance)).unsqueeze(-1)

    
    def set_ls(self,ls:float):
        '''便于求应力强度因子设置ls为定值'''
        def constant(xy):
            x = xy[...,0]; y = xy[...,1]
            return self.one(x,y) * ls

        self.getH = constant

    def restore_ls(self):
        self.getH = self.getH_standard
        
    
class LineCrackEmbedding(CrackEmbedding):
    def __init__(self, xy0, xy1, tip, decay_alpha: float = 10.0):
        self.Line = LineSegement(xy0,xy1)
        super().__init__(levelset = self.Line.levelset_dist,
                         xy0=xy0, xy1=xy1, tip=tip,
                         left_beta = self.Line.tangent_theta + np.pi,
                         right_beta = self.Line.tangent_theta,
                         decay_alpha = decay_alpha,
                         )
        

class multiLineCrackEmbedding(Embedding):
    def __init__(self, points: list[list], tip='both', b=0.5,
                 decay_alpha: float = 10.0, decay_p: float = 1.0, decay_ell: float | None = None):
        self.xy0 = points[0]
        self.xy1 = points[-1]
        self.MAX_X = b
       # if self.xy0[-1] > self.MAX_X: raise Exception('Please change self.MAX_X')

        if tip == 'right':
            inner_points = points + [[self.MAX_X, points[-1][1]]]
            self.inner_surfaces = multiInnerSurfaces(
                inner_points, decay_alpha=decay_alpha, decay_p=decay_p, decay_ell=decay_ell
            )
            self.right_tip = LineCrackEmbedding(points[-2], points[-1], tip='right', decay_alpha=decay_alpha)
            self.split_line = LineSegement.init_theta(points[-2], self.right_tip.Line.tangent_theta + np.pi/2)
        else:
            raise Exception('Not Implemented!')

        self.set_Heaviside(0)

    def getGamma(self, xy) -> torch.Tensor:
        left_H = self.Heaviside(self.split_line.levelset(xy[...,0], xy[...,1])).unsqueeze(-1)
        inner_basis = self.inner_surfaces.getGamma(xy) * (1 - left_H)
        right_tip   = self.right_tip.getGamma(xy) * left_H
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
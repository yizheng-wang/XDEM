import torch
import numpy as np


class LocalAxis:
    def to_tensor(self,x):
        if torch.is_tensor(x):
            return x
        else:
            return torch.tensor(x)

    def __init__(self,x0,y0,beta=0.0):
        self.x0 = self.to_tensor(x0)
        self.y0 = self.to_tensor(y0)
        self.beta = self.to_tensor(beta)
        self.cos = torch.cos(self.beta)
        self.sin = torch.sin(self.beta)

        self.Q = torch.tensor([[self.cos,self.sin],[-self.sin,self.cos]])

    def cartesianToPolar(self,x,y):
        '''整体直角坐标转换为裂尖的局部极坐标'''
        r = self.getR(x,y)
        theta = self.getLocalTheta(x,y)
        return r,theta
    
    def getR(self,x,y):
        return torch.sqrt( (x-self.x0) ** 2 + (y-self.y0) ** 2 )

    def getLocalTheta(self,x,y):
        local_x,local_y = self.cartesianToLocal(x,y)
        theta = torch.arctan2(local_y,local_x)
        return theta
    
    def cartesianVariableToLocal(self,f_x,f_y):
        '''将直角坐标的物理量转换为局部坐标'''
        f_local_x = self.cos * f_x + self.sin * f_y
        f_local_y = - self.sin * f_x + self.cos * f_y
        return f_local_x , f_local_y     
    
    def cartesianToLocal(self,x,y):
        '''相对于裂纹尖端方向的xy坐标'''
        return self.cartesianVariableToLocal(x-self.x0 , y-self.y0)
    
    def LocalToCartesian(self,local_x,local_y):
        '''坐标系反向旋转为全局坐标系'''
        global_x = self.cos * local_x - self.sin * local_y
        global_y = self.sin * local_x + self.cos * local_y
        '''平移原点'''
        x = global_x + self.x0
        y = global_y + self.y0      
        return x , y  
    
    def polarToCartesian(self,r,theta):
        '''相对于裂纹尖端方向的xy局部坐标'''
        local_x = r * torch.cos(theta)
        local_y = r * torch.sin(theta)
        return self.LocalToCartesian(local_x,local_y)
    
    def tensorTolocal(self,X:torch.Tensor):
        '''将二阶张量旋转至局部坐标系'''
        return self.Q.unsqueeze(0) @ X @ self.Q.T.unsqueeze(0)


class Geometry1D:
    def to_tensor(self,x):
        if torch.is_tensor(x):
            return x
        else:
            return torch.tensor(x)

    def get_tangent_theta(self,x,y)->torch.Tensor:...
    def get_normal_theta(self,x,y)->torch.Tensor:
        '''线段法线方向与x轴的夹角'''
        return self.get_tangent_theta(x,y) + torch.pi/2
    
    def get_direction_cosine(self,x,y)->torch.Tensor:
        normal_theta = self.get_normal_theta(x,y)
        l_x = torch.cos(normal_theta)
        l_y = torch.cos(torch.pi/2 - normal_theta)
        return l_x,l_y
    
    def transform_to_surface(self,x,y,f_11,f_22,f_12):
        l_x , l_y = self.get_direction_cosine(x,y)
        fx = l_x * f_11 + l_y * f_12
        fy = l_x * f_12 + l_y * f_22
        return fx , fy

    def generate_random_points(self,num)->torch.Tensor:...
    def generate_linespace_points(self,num)->torch.Tensor:...
    def levelset(self,x,y):...
    def is_on_geometry(self,points,eps=1e-4):...
    
    def is_on_left(self, points):
        x = points[...,0]
        y = points[...,1]
        ls = self.levelset(x,y)
        return torch.where(ls < 0 , True, False)

class LineSegement(Geometry1D):
    def __init__(self,xy0,xy1) -> None:
        self.x0 = self.to_tensor(xy0[0])
        self.x1 = self.to_tensor(xy1[0])
        self.y0 = self.to_tensor(xy0[1])
        self.y1 = self.to_tensor(xy1[1])
        self.x_span = [self.x0,self.x1]
        self.y_span = [self.y0,self.y1]
        self.x_len = self.x1 - self.x0
        self.y_len = self.y1 - self.y0
        self.tangent_theta = torch.arctan2(self.y_len,self.x_len)
        self.A = self.y1 - self.y0
        self.B = self.x0 - self.x1
        self.C = self.x1 * self.y0 - self.x0 * self.y1
        self.AB2 = self.A **2 + self.B** 2
        self.length = torch.sqrt((self.x0 - self.x1)**2 + (self.y0 - self.y1)**2)
    
    @classmethod
    def init_theta(line,xy0,tan_beta):
        x1 = xy0[0] + np.cos(tan_beta)
        y1 = xy0[1] + np.sin(tan_beta)
        return line(xy0,[x1,y1])

    def levelset(self,x,y):
        return (self.A * x + self.B * y + self.C) / torch.sqrt(self.AB2) # 计算点到直线的距离
    
    def levelset_dist(self,x,y):
        return (self.A * x + self.B * y + self.C) / torch.sqrt(self.AB2) # 计算点到直线的距离
    
    def get_tangent_theta(self, x, y):
        return self.tangent_theta * torch.ones_like(x)
    
    def generate_random_points(self,num):
        random = np.random.rand(num)
        x = random * self.x_span[0].numpy() + (1 - random) * self.x_span[1].numpy()
        y = random * self.y_span[0].numpy() + (1 - random) * self.y_span[1].numpy()
        return torch.tensor(x) , torch.tensor(y)
    
    def generate_linespace_points(self, num) :
        x = torch.linspace(self.x_span[0],self.x_span[1],num)
        y = torch.linspace(self.y_span[0],self.y_span[1],num)
        return x,y
    
    def approx_dist(self,x,y):
        def dist(x1,x2,y1,y2):
            return torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)    
         
        dist_0 = dist(self.x0,x,self.y0,y)
        dist_1 = dist(self.x1,x,self.y1,y)   
        return (dist_0 + dist_1 - self.length)
    
    def is_on_geometry(self, points, eps=1e-4):

        x = points[...,0]
        y = points[...,1]

        return torch.where( self.approx_dist(x,y)< eps, True, False)
    
    def clamp(self,ratio = None,
                   dist1 = None,
                   dist2 = None):

        if ratio is not None:
            dist = ratio * self.length
        elif dist1 is not None:
            dist = dist1
        elif dist2 is not None:
            dist = self.length - dist2

        dist_x = dist * np.cos(self.tangent_theta)
        dist_y = dist * np.sin(self.tangent_theta)

        return self.x0 + dist_x, self.y0 + dist_y


class MultiSegement1D(Geometry1D):
    def __init__(self,Geo_list:list[LineSegement]) -> None:
        super().__init__()
        self.geometries = Geo_list

    
    def levelset(self,x,y)->torch.Tensor:
        levelsets = torch.stack(list(map(lambda ls: ls.levelset_dist(x,y),self.geometries)))
        distances = torch.stack(list(map(lambda d: d.approx_dist(x,y),self.geometries)))
        index = torch.argmin(torch.abs(distances),dim=0)
        ls =  levelsets[index, torch.arange(len(index))]
        return ls


class Circle:

    def __init__(self,x0,y0,r) -> None:
        self.local_axis = LocalAxis(x0,y0)
        self.r = r
        self.area = np.pi * r **2
    
    def dist(self,x,y):
        return self.local_axis.getR(x,y)

    def levelset(self,x,y):
        return self.dist(x,y) - self.r
    
    def is_in_geometry(self,x,y):
        return self.levelset(x,y) < 0

    def generate_random_points(self,num):
        random = torch.from_numpy(np.random.rand(num))
        theta = random * torch.pi * 2
        r = random * self.r
        x,y = self.local_axis.polarToCartesian(r,theta)
        return x,y

    def generate_arc_points(self, num, init_beta) :
        '''在圆弧上生成等角度间隔排列的点'''

        theta = torch.linspace(-torch.pi+1e-4, torch.pi-1e-4,num)
        y = torch.linspace(self.y_span[0],self.y_span[1],num)
    
    def area_in_rect(self,left ,right ,bottom ,top):

        x_circle,y_circle = self.generate_random_points(10000)
        is_whole_in_rect = torch.all((x_circle > left) & (x_circle < right) & (y_circle > bottom) & (y_circle < top))
        if  is_whole_in_rect:
            return self.area
        else:
            '''蒙特卡洛积分计算矩形与圆的重叠面积'''
            square = (right - left) * (top - bottom)
            num_points = 100000000
            x = torch.from_numpy(np.random.uniform(left, right, size=(num_points)))
            y = torch.from_numpy(np.random.uniform(bottom, top, size=(num_points)))
            mask = self.is_in_geometry(x,y)
            points_inside_circle = torch.sum(mask).numpy()
            overlap_area = (points_inside_circle / num_points) * square
            return overlap_area

    def norm_dist(self,x,y):
        return self.dist(x,y) / self.r

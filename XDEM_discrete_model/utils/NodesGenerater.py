import torch
import numpy as np
import utils.Geometry as Geometry

def meshgirdFromXY(x,y):

    X,Y = torch.meshgrid(x, y)     
    X=X.reshape(-1)
    Y=Y.reshape(-1)
    return X,Y    

def genMeshNodes2D(xstart,xend,xnum,ystart,yend,ynum):
    '''生成规则网格排列的二维点'''
    x = torch.linspace(xstart,xend,xnum)
    y = torch.linspace(ystart,yend,ynum)

    return meshgirdFromXY(x,y)

def genRandomNodes2D(left, right, bottom, top, num):
    '''生成服从均匀分布的随机二维点'''
    random = np.random.rand(num)
    x = random * left + (1 - random) * right
    random = np.random.rand(num)
    y = random * bottom + (1 - random) * top
    return torch.stack([torch.tensor(x) , torch.tensor(y)],dim=1)

def genHeteroNodes1D(sep:list,num:list):
    '''从给定加密区域生成一维局部均匀分布点'''
    region_num = len(sep) - 1
    if region_num != len(num):
        raise Exception()
    node_set = [torch.linspace(sep[i],sep[i+1],num[i]) for i in range(region_num)]
    return torch.cat(node_set,dim=-1)

def genHeteroNodes2D(xsep,xnum,ysep,ynum):
    x = genHeteroNodes1D(xsep,xnum)
    y = genHeteroNodes1D(ysep,ynum)
    return meshgirdFromXY(x,y)

def genHeteroTip2D(xstart,xend,
                    ystart,yend,
                    xTip,yTip,
                    x_dense_num,y_dense_num,
                    x_outer_num,y_outer_num,
                    x_inteval=0.2,y_inteval=0.2):
    
    def segmentTip(start,end,Tip,
                   dense_num,outer_num,
                   inteval):
        # length = end - start - 2*inteval
        Tip_left = Tip - inteval
        Tip_right = Tip + inteval
        length = end - start - min([inteval,Tip - start]) - min([end - Tip,inteval])
        segments = [start]
        num = []
        if Tip_left - start > 0.01:
            segments.append(Tip_left)
            left_num = round(outer_num * (Tip_left - start)/length)
            num.append(left_num)
        else: left_num = 0

        num.append(dense_num)
        if end - Tip_right > 0.01:
            segments.append(Tip_right)
            num.append(outer_num-left_num)
        segments.append(end)

        return segments,num
    
    xsep,xnum = segmentTip(xstart,xend,xTip,x_dense_num,x_outer_num,x_inteval)

    ysep,ynum = segmentTip(ystart,yend,yTip,y_dense_num,y_outer_num,y_inteval)

    return meshgirdFromXY(genHeteroNodes1D(xsep,xnum),genHeteroNodes1D(ysep,ynum))


def DeleteTips(points,pdf,axes:list[Geometry.LocalAxis],delta = 1e-2):
    for i,axis in enumerate(axes):
        r = axis.getR(points[...,0],points[...,1])
        points = points[r>delta]
        pdf = pdf[r>delta]
    return points,pdf

def genDenseCircles(left, right, bottom, top, outer_num,
                    circles:list[Geometry.Circle],
                    inner_num):
    '''加密多个裂尖周围的积分点'''

    points_x=[]
    points_y=[]
    points_pdf = []

    circles_area = 0.0
    total_circle_num = 0

    outer_points = genRandomNodes2D(left, right, bottom, top,
                num = int(outer_num * (right - left) * (top - bottom)/ ((right - left) * (top - bottom) - sum([circle.area for circle in circles]))))
    
    # 用于记录在圆内的外部点的索引
    mask_outer = torch.zeros_like(outer_points[...,0],dtype=torch.bool)
    for i,circle in enumerate(circles):

        rtheta = genRandomNodes2D(2e-4, circle.r, 0, 2*np.pi, inner_num)
        x,y = circle.local_axis.polarToCartesian(rtheta[...,0],rtheta[...,1])

        # 筛选掉不在矩形域内的点
        circle_area = circle.area_in_rect(left, right, bottom, top)
        circles_area += circle_area

        ind = torch.where((x > left) & (x < right) & (y > bottom) & (y < top))
        x = x[ind]
        y = y[ind]
        rtheta = rtheta[ind]

        points_x.append(x) ; points_y.append(y)

        total_circle_num += x.data.nelement()

        # I=int(f(r,theta)*r*drdtheta
        points_pdf.append(torch.ones_like(x) / (rtheta[...,0] * (2*np.pi * circle.r))) 

        mask_outer |= circle.is_in_geometry(outer_points[...,0],outer_points[...,1])

    outer_pdf = torch.ones_like(outer_points[...,0]) / ((right - left) * (top - bottom) - circles_area)    
    outer_points = outer_points[~mask_outer]

    outer_num = outer_points[...,0].data.nelement()
    total_num = outer_num + total_circle_num    
    outer_pdf = outer_pdf[~mask_outer] * outer_num / (total_num)

    for point_pdf in points_pdf : point_pdf *= inner_num / (total_num)

    points_x.append(outer_points[...,0])
    points_y.append(outer_points[...,1])
    points_pdf.append(outer_pdf)

    points_x = torch.cat(points_x)
    points_y = torch.cat(points_y)
    points_pdf = torch.cat(points_pdf)
    return torch.stack((points_x,points_y),dim=1) , points_pdf.numpy()

def gen1DLine(start,end,num):
    x_uniform = np.arange(0.001, 1, 1/num)
    x_transform = np.sqrt(x_uniform)
    x = x_transform * (end - start) + start
    return torch.from_numpy(x)

def genTipDense1D(start,end,Tip,num):
    left_num = int((Tip - start) / (end - start) * num)
    right_num = num - left_num
    x_left = gen1DLine(start,Tip,left_num)
    x_right = gen1DLine(end,Tip,right_num).flip(dims=[-1])
    return torch.cat((x_left,x_right),dim=-1)

def genTipDenseMesh(xstart,xend,
                    ystart,yend,
                    xTip,yTip,
                    xnum,ynum):
    x = genTipDense1D(xstart,xend,xTip,xnum)
    y = genTipDense1D(ystart,yend,yTip,ynum)
    return meshgirdFromXY(x,y)
    
    

import torch

def get_grad(y,x)->torch.Tensor:
    '''计算单个y对x的偏导数'''
    return torch.autograd.grad(inputs = x, 
                               outputs = y, 
                               grad_outputs=torch.ones_like(y), 
                               retain_graph=True, 
                               create_graph=True)[0]


if __name__ == '__main__':
    '''这些是求导的注意事项，
    切片索引后不能继承原先对象的导数'''
    x = torch.tensor(0.6).requires_grad_()
    y = x - 0.5
    xy = torch.stack((x,y),dim=0)
    l = xy[0] + xy[1]
    xyl = torch.stack((xy[...,0],xy[...,1],l),dim=0)
    z = xyl[0] + xyl[1]+ xyl[2]
    # z = torch.abs(y)
    # print(get_grad(y,x))
    # print(get_grad(-y,x))
    print(get_grad(l,xy))
    print(get_grad(z,xy))
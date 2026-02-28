import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.get_grad import get_grad
from utils.EarlyStopping import EarlyStopping
from utils.Integral import montecarlo,trapz2D,simps2D
from utils.NodesGenerater import genMeshNodes2D,genHeteroTip2D
from utils.NN import weight_init
import matplotlib.pyplot as plt
from matplotlib import colors,cm
from utils.Geometry import Geometry1D
import matplotlib.ticker as ticker
import gc


#Pa,N,m
#模型输出位移是mm
#输出应力单位MPa
class PINN2D:
    def __init__(self,model:nn.Module):
        '''ONLY allow 'self.XY' for forward computation  besause of the map for autodiff'''
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []

        self.model.apply(weight_init)    #初始化权重
        self.integral = montecarlo
        self.E_int = self.E_int_montecarlo
        self.params = self.model.parameters()

    def clear(self):
        del self.model
        del self.X
        del self.Y
        del self.XY
        del self.optimizer
        del self.losses
        # del self.params
        gc.collect()
        torch.cuda.empty_cache()  # 手动释放未使用的缓存显存

    def variable(self,x:torch.Tensor):
        return x.float().requires_grad_().to(device=self.device)
    
    def _set_points(self,x,y)->torch.Tensor:
        x = self.variable(x)
        y = self.variable(y)
        xy = torch.stack([x,y],dim=1)
        return x , y , xy

    def set_inner_points(self,internal_points,internal_points_pdf: torch.Tensor,variable = True):
        '''设置域内的采样点'''

        if variable:
            self.X,self.Y,self.XY = self._set_points(internal_points[:,0] , internal_points[:,1])
        else:
            self.X,self.Y,self.XY = self._set_points(internal_points[0] , internal_points[1])

        if type(internal_points_pdf) is np.ndarray:
            self.points_pdf = torch.from_numpy(internal_points_pdf).to(device=self.device)
        elif torch.is_tensor(internal_points_pdf):
            self.points_pdf = internal_points_pdf.to(device=self.device)
        else:
            raise Exception('Type error!')
        self.zero=torch.zeros_like(self.X)  
    
    ### physics ###

    def setMaterial(self,E,nu,type='plane stress'):
        '''杨氏模量单位为MPa'''
        self.E = E
        self.mu = nu
        self.setD(self.E,self.mu,type=type)

    def setD(self,E,mu,type='plane stress'):
        if type=='plane stress':
            self.d11 = torch.tensor(E/(1-mu*mu)).to(self.device)
            self.d12 = torch.tensor(E/(1-mu*mu)*mu).to(self.device)
            self.G = torch.tensor(E/(2*(1+mu))).to(self.device)
        elif type == 'plane strain':
            self.d11 = torch.tensor(E*(1-mu)/((1+mu)*(1-2*mu))).to(self.device)
            self.d12 = torch.tensor(E*mu/((1+mu)*(1-2*mu))).to(self.device)
            self.G = torch.tensor(E/(2*(1+mu))).to(self.device)

        else: raise Exception('error!')    

    def hard_u(self,u,x,y):
        return u
    def hard_v(self,v,x,y):
        return v

    def pred_uv(self,xy):
        uv = self.model(xy)
        u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])      
        return u,v  
    
    def compute_Strain(self,u,v,xy):
        du_dxy = get_grad(u,xy)
        dv_dxy = get_grad(v,xy)
        eXX = du_dxy[...,0]
        eYY = dv_dxy[...,1]
        eXY = dv_dxy[...,0] + du_dxy[...,1]

        return eXX,eYY,eXY
    
    def constitutive(self,eXX,eYY,eXY):
        sx = self.d11 * eXX + self.d12 * eYY
        sy = self.d12 * eXX + self.d11 * eYY
        sxy = self.G * eXY
        return sx , sy , sxy
    
    def Equilibrium_loss(self)->torch.Tensor:
        '''no body force'''
        sx,sy,sxy = self.pred_stress(self.XY)
        dsx_dx = get_grad(sx,self.X)
        dsy_dy = get_grad(sy,self.Y)
        dsxy_dxy = get_grad(sxy,self.XY)
        dsxy_dx , dsxy_dy = dsxy_dxy[...,0] , dsxy_dxy[...,1]
        fx_loss = self.criterion(dsx_dx + dsxy_dy,self.zero)
        fy_loss = self.criterion(dsy_dy + dsxy_dx,self.zero)
        return torch.stack([fx_loss,fy_loss]) 
    
    def get_energy_density(self,xy):
        u,v = self.pred_uv(xy)
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        sx,sy,sxy = self.constitutive(eXX,eYY,eXY)
        energy =  0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        return energy

    def E_int_montecarlo(self)->torch.Tensor:
        energy = self.get_energy_density(self.XY)
        return montecarlo(energy , self.points_pdf)
    
    def E_ext(self)->torch.Tensor:
        return torch.tensor(0.0,device=self.device)

    def Energy_loss(self)->torch.Tensor:
        return torch.stack([self.E_int() - self.E_ext()])
    
    def soft_BC_loss(self)->torch.Tensor:
        return torch.tensor(0.0,device=self.device)
    
    def pred_stress(self,xy):
        _,_,sx,sy,sxy = self.infer(xy)
        return sx,sy,sxy    

    def infer(self,xy):
        u,v = self.pred_uv(xy) 
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        sx , sy , sxy = self.constitutive(eXX,eYY,eXY)   
        return u,v,sx,sy,sxy

    def stressToMises(self,sx,sy,sxy):
        return torch.sqrt((sx-sy)**2/2 + 3*sxy**2)
    
    def displacement(self,u,v):
        return torch.sqrt(u**2+v**2)

    ### Sampling ###

    def set_meshgrid_inner_points(self,xstart,xend,xnum,ystart,yend,ynum):
        '''生成规则网格排布点'''
        x,y = genMeshNodes2D(xstart,xend,xnum,ystart,yend,ynum)
        self.X,self.Y,self.XY = self._set_points(x,y)
        self.points_pdf = 1 / ((xend - xstart) * (yend - ystart))
        self.xshape = xnum
        self.yshape = ynum
        def E_int_trapz():
            energy = self.get_energy_density(self.XY)
            return trapz2D(energy , self.XY , [self.xshape,self.yshape])
        self.E_int = E_int_trapz
        self.zero = torch.zeros_like(self.X) 

    def set_meshgrid_trapz_Tip_Dense(self,xstart,xend,
                    ystart,yend,
                    xTip,yTip,
                    x_dense_num,y_dense_num,
                    x_outer_num,y_outer_num,
                    x_inteval=0.2,y_inteval=0.2):
        '''生成规则网格排布点'''
        x,y = genHeteroTip2D(xstart,xend,
                            ystart,yend,
                            xTip,yTip,
                            x_dense_num,y_dense_num,
                            x_outer_num,y_outer_num,
                            x_inteval=x_inteval,
                            y_inteval=y_inteval)
        self.X,self.Y,self.XY = self._set_points(x,y)
        self.xshape = x_dense_num + x_outer_num
        self.yshape = y_dense_num + y_outer_num
        def E_int_trapz():
            energy = self.get_energy_density(self.XY)
            return trapz2D(energy , self.XY , [self.xshape,self.yshape])
        self.E_int = E_int_trapz
        self.zero = torch.zeros_like(self.X) 

    def set_meshgrid_simps_points(self,xstart,xend,x_interval,
                                  ystart,yend,y_interval):
        xnum = x_interval *2 -1
        ynum = y_interval *2 -1
        self.set_meshgrid_inner_points(xstart,xend,xnum,ystart,yend,ynum)
        def E_int_simps():
            energy = self.get_energy_density(self.XY)
            return simps2D(energy , self.XY , [self.xshape,self.yshape])
        self.E_int = E_int_simps


    def add_BCPoints():...

    ### Train ###

    def pde_loss(self)->torch.Tensor:...
    
    def set_loss_func(self,losses:list,weights = None):
        self.losses = losses
        if weights is None:
            self.weights = torch.Tensor([1.0]*len(losses)).to(self.device)
        else:
            self.weights = torch.Tensor(weights).to(self.device)


    def get_loss(self) -> torch.Tensor:
        loss_array = torch.cat(list(map(lambda x: x(),self.losses)))
        loss_sum = torch.sum(self.weights * loss_array)
        return loss_array , loss_sum

    def __loss_func(self) -> torch.Tensor:
        loss_array , loss_sum = self.get_loss()
        loss_sum.backward()
        return loss_sum

    def print_loss(self):
        print(self.iter,":",end=" ")
        strain_energy = self.E_int().cpu().detach().numpy()
        external_work = self.E_ext().cpu().detach().numpy()
        for loss in self.losses:
                print(loss.__name__,':',
                      loss()[0].cpu().detach().numpy(),":",end=" ")
                print(f'Strain_Energy = {strain_energy:.3e} | External_Work = {external_work:.3e}')

    def eval(self):
        loss_array,loss_sum = self.get_loss()
        self.print_loss()
        self.record()
        self.EarlyStopping(loss_sum.cpu().detach().numpy(),self.model)      #判断是否需要提前结束训练  

    def record(self):
        item = self.record_item()
        self.history.append(item)     

    def record_item(self):  
        return  self.E_int().cpu().detach().numpy(), self.E_ext().cpu().detach().numpy(),self.Energy_loss().cpu().detach().numpy()[0]

    def train_step(self):
        self.optimizer.zero_grad()
        self.optimizer.step(self.__loss_func)
        self.iter = self.iter + 1 

        if self.iter % 10000 == 0: 
            self.save(self.path+str(self.iter))
            self.save_hist(self.path)

    def set_Optimizer(self,lr):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr= lr)
    
    def set_EarlyStopping(self,patience,verbose, path):
        self.EarlyStopping=EarlyStopping(patience,verbose=verbose,path=path+'.pth')

    def train(self,epochs = 50000, patience=10 , path = 'test', lr= 0.02,eval_sep=100,
              milestones=np.arange(20000,50000,20000), tol_early_stop=0.01):

        self.iter = 0
        self.set_EarlyStopping(patience=patience,verbose=True,path= path)
        self.path = path
        self.set_Optimizer(lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma = 0.5)

        for i in range(epochs):

            self.train_step()

            if self.iter % eval_sep == 0: 
                self.eval()
                                # Calculate strain energy and external work separately


            scheduler.step()
            if (self.EarlyStopping.early_stop):
                print('end epoch:'+str(self.iter))
                break
        self.save_hist(self.path)

    def save_hist(self,name):
        hist = pd.DataFrame(self.history)
        hist.to_csv(name+'.csv')

    def save(self,name):
        self.save_hist(name)
        torch.save(self.model.state_dict(),name+'.pth')

    def load(self,path,loadtype='state'):
        if loadtype=='state':
            self.model.load_state_dict(torch.load(path+'.pth',map_location=torch.device('cpu')))
        else:
            self.model=torch.load(path+'.pth',map_location=torch.device('cpu'))

    ### Postprocessing ###

    def readData(self,path):
        '''读取comsol结果'''
        self.labeled=True
        df=pd.read_csv(path,skiprows=9,names=['x','y','u','v','sx','sy','sxy'],delim_whitespace=True)  #分割为空白字符
        # df = df.sample(9)
        self.labeled_x = torch.tensor(df['x'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_y = torch.tensor(df['y'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_xy = torch.stack([self.labeled_x,self.labeled_y],dim=1)
        self.labeled_u = torch.tensor(df['u'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_v = torch.tensor(df['v'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sx = torch.tensor(df['sx'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sy = torch.tensor(df['sy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)       
        self.labeled_sxy = torch.tensor(df['sxy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)

    def  plotContourf(self,X,Y,Z,ax = plt.subplot(1,1,1),vmax = None,show = False,colorbar_norm = None,cmap = 'jet', cbar=True,levels=200)->plt.Axes:

        nx = np.unique(X).size
        ny = np.unique(Y).size

        X_grid = X.reshape(nx, ny)
        Y_grid = Y.reshape(nx, ny)
        Z_grid = Z.reshape(nx, ny)


        if vmax:
            plot = ax.contourf(X_grid,Y_grid,Z_grid,cmap=cmap,levels=np.linspace(vmax[0],vmax[1],50),extend='both')
        elif colorbar_norm is not None:
            plot = ax.contourf(X_grid,Y_grid,Z_grid,levels,cmap=cmap,norm = colorbar_norm,extend=False)
        else:
            plot = ax.contourf(X_grid,Y_grid,Z_grid,levels,cmap=cmap)
        if cbar:
            cb = self.plot_cbar(ax=ax,plot=plot,colorbar_norm=colorbar_norm,cmap=cmap)


        ax.axis('equal')
        ax.autoscale(tight=True)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
        if show:
            plt.show()
            return
        return plot
    
    def plot_cbar(self,ax,plot,cmap='jet',colorbar_norm=None,cax=None,shrink=1.0):

        if colorbar_norm is not None:
            cb = plt.colorbar(cm.ScalarMappable(norm=colorbar_norm, cmap=cmap), ax=ax, cax=cax,shrink=shrink,extend = 'max')
        else:
            cb = plt.colorbar(plot,ax=ax,cax=cax,shrink=shrink)

        # 获取 colorbar 的数据范围
        vmin, vmax = cb.vmin, cb.vmax
        exponent = int(np.floor(np.log10(max(abs(vmin), abs(vmax)))))
        if exponent >-2 : exponent = 0

        # 自定义格式
        def format_func(value, pos):
            return f'{value / 10**exponent:.1f}'  # 保留两位小数

        n_extra = 1  
        ticks_middle = np.linspace(vmin, vmax, n_extra + 2)[1:-1]  
        ticks = np.concatenate(([vmin], ticks_middle, [vmax]))

        cb.set_ticks(ticks)

        
        if exponent != 0:
            cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            cb.ax.annotate(f'$\\times 10^{{{int(exponent)}}}$', 
                            xy=(3.0, 1.01), xycoords='axes fraction', 
                            ha='center', va='bottom')

        return cb

    def plot_error(self,x,y,z,z_refer,title='output',name = None,levels=200):
        import matplotlib.pyplot as plt
        abs_error = np.abs((z - z_refer))
        norm_abs_error = np.abs((z - z_refer)/(np.max(z_refer)))
        relative_error = np.abs(z - z_refer)/(np.abs(z_refer)+1e-3*np.max(np.abs(z_refer)))       
        
        norm = colors.Normalize(vmin=np.min(z_refer), vmax=np.max(z_refer))

        ax = plt.subplot(2,2,2)
        plot = self.plotContourf(x,y,z_refer,ax,colorbar_norm=norm,cmap='jet',levels=levels)
        ax.set_title('FEM')

        bx = plt.subplot(2,2,1)
        plot = self.plotContourf(x,y,z,bx,colorbar_norm=norm,cmap='jet',levels=levels)
        bx.set_title('DENNs') 
               
        ax = plt.subplot(2,2,3)        
        plot = self.plotContourf(x,y,norm_abs_error,ax,levels=levels)   
        ax.set_title('normed abs error')

        ax = plt.subplot(2,2,4)
        vmax = [0.0,0.2] if np.max(relative_error) > 0.5 else False
        plot = self.plotContourf(x,y,relative_error,ax,vmax,levels=levels)   
        ax.set_title('relative error') 
        plt.suptitle(title,fontsize = 15)  
        plt.tight_layout()  
        
        if name is None:
            plt.show()
        else:
            plt.savefig(name+title+'.jpg', dpi=300)
    
    def rmse(self,z,z_refer):
        mse = (z - z_refer) **2
        z_refer_square = z_refer**2
        return np.sqrt(np.sum(mse)/np.sum(z_refer_square))

    def showPrediction(self,xy):
        u,v = self.pred_uv(xy)
        u=u.cpu().detach().numpy()
        v=v.cpu().detach().numpy()
        x=xy[...,0].cpu().detach().numpy()
        y=xy[...,1].cpu().detach().numpy()
        self.plotContourf(x,y,u,show=True)
        plt.title('u')
        plt.figure()
        self.plotContourf(x,y,v,show=True,ax=plt.subplot(1,1,1))
        plt.title('v')
        sx,sy,sxy = self.pred_stress(xy)
        sx = sx.cpu().detach().numpy()
        sy = sy.cpu().detach().numpy()
        sxy = sxy.cpu().detach().numpy()
        plt.figure()
        self.plotContourf(x,y,sx,show=True,ax=plt.subplot(1,1,1))
        plt.title('sx')
        plt.figure()
        self.plotContourf(x,y,sy,show=True,ax=plt.subplot(1,1,1))
        plt.title('sy')
        plt.figure()
        self.plotContourf(x,y,sxy,show=True,ax=plt.subplot(1,1,1))
        plt.title('sxy')


    def evaluate(self,name = None,levels=200):
        u,v = self.pred_uv(self.labeled_xy)
        u=u.cpu().detach().numpy()
        v=v.cpu().detach().numpy()

        x=self.labeled_x.cpu().detach().numpy()
        y=self.labeled_y.cpu().detach().numpy()
        u_refer = self.labeled_u.cpu().detach().numpy()
        v_refer = self.labeled_v.cpu().detach().numpy()
        self.plot_error(x,y,u,u_refer,title='u',name=name,levels=levels)
        self.plot_error(x,y,v,v_refer,title='v',name=name,levels=levels)
        sx,sy,sxy = self.pred_stress(self.labeled_xy)
        sx = sx.cpu().detach().numpy()
        sy = sy.cpu().detach().numpy()
        sxy = sxy.cpu().detach().numpy()
        sx_refer = self.labeled_sx.cpu().detach().numpy()
        sy_refer = self.labeled_sy.cpu().detach().numpy()
        sxy_refer = self.labeled_sxy.cpu().detach().numpy()
        self.plot_error(x,y,sx,sx_refer,title='sx',name=name,levels=levels)
        self.plot_error(x,y,sy,sy_refer,title='sy',name=name,levels=levels)
        self.plot_error(x,y,sxy,sxy_refer,title='sxy',name=name,levels=levels)    




class PINN2D_bimaterial(PINN2D):

    def set_LevelSet(self,line:Geometry1D):
        self.material_interface = line


    def setMaterial(self, E1, E2, nu1 = 0.3, nu2 = 0.3 ,type='plane stress'):
        '''下标1为material_interface左侧,2为右侧'''
        self.E = np.array([E1,E2])
        self.nu = np.array([nu1,nu2])
        self.setD(self.E , self.nu,type=type)
    
    
    def set_inner_materials(self):
        d11 , d12 , G = self.get_material(self.XY)
        self.inner_d11 = d11
        self.inner_d12 = d12
        self.inner_G = G

    
    def get_material(self,xy):
        ls  = self.material_interface.levelset(xy[...,0],xy[...,1])
        ind = torch.where(ls < 0 , 0, 1)
        d11 = self.d11[ind]
        d12 = self.d12[ind]
        G   = self.G[ind]
        return d11 , d12 , G    
    
    def pred_stress(self, xy):
        u,v = self.pred_uv(xy) 
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        d11 , d12 , G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY , d11 , d12 , G)
        return sx, sy, sxy
    
    def infer(self, xy):
        u,v = self.pred_uv(xy) 
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        d11 , d12 , G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY , d11 , d12 , G)
        return u,v,sx, sy, sxy
    
    def constitutive(self, eXX, eYY, eXY , d11 , d12 , G):
        sx = d11 * eXX + d12 * eYY
        sy = d12 * eXX + d11 * eYY
        sxy = G * eXY
        return sx , sy , sxy

    def get_energy_density(self,xy):
            u,v = self.pred_uv(xy)
            eXX,eYY,eXY = self.compute_Strain(u,v,xy)
            d11,d12,G = self.get_material(xy)
            sx,sy,sxy = self.constitutive(eXX,eYY,eXY,d11,d12,G)
            energy = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
            return energy
      

    def set_meshgrid_inner_points(self,xstart,xend,xnum,ystart,yend,ynum):
        '''生成规则网格排布点'''
        super().set_meshgrid_inner_points(xstart,xend,xnum,ystart,yend,ynum)
        self.set_inner_materials()

    def set_meshgrid_trapz_Tip_Dense(self, xstart, xend, ystart, yend, xTip, yTip, 
                                     x_dense_num, y_dense_num, x_outer_num, y_outer_num, 
                                     x_inteval=0.2,y_inteval=0.2):
        super().set_meshgrid_trapz_Tip_Dense(xstart, xend, ystart, yend, xTip, yTip, 
                                             x_dense_num, y_dense_num, x_outer_num, y_outer_num, 
                                             x_inteval , y_inteval)
        self.set_inner_materials()
    

        

    

import torch.nn as nn
import torch

def weight_init(m):
    '''初始化模型参数'''
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

class Block(nn.Module):
    def __init__(self,input_shape,width,activation=nn.ReLU) -> None:
        super().__init__()

        self.layers=nn.Sequential(nn.Linear(input_shape,width),activation(),
                                  nn.Linear(width,input_shape),activation())
    def forward(self,x):
        return self.layers(x)+x


class ResidualNet(nn.Module):
    def __init__(self,input=2,output=2,width=50,activation=nn.ReLU,hidden_layer_num=4) -> None:
        super().__init__()
        self.activation = activation()

        self.input_layer = nn.Linear(input, width)

        self.block_layers = nn.ModuleList()
        for i in range(hidden_layer_num):
            self.block_layers.append(Block(width, width,activation))

        self.output_layer = nn.Linear(width, output)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.block_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
    
class stack_net(nn.Module):
    def __init__(self,input=2,output=2,width=50,activation=nn.Tanh,net = ResidualNet,depth = 4) -> None:
        super().__init__()
 
        for i in range(output):
            setattr(self, "tower"+str(i+1), net(input=input,output=1,width=width,activation=activation,hidden_layer_num=depth)) 
        self.towers = [getattr(self,"tower"+str(i+1)) for i in range(output)] 

    def forward(self,x):
        return [model(x) for model in self.towers]
    

class MultilayerNN(nn.Module):
    def __init__(self, width, hidden_layer_num =4, activation=nn.Tanh,  input =2, output=1):
        super(MultilayerNN, self).__init__()
        self.activation = activation()
        if type(width) == int:
            width = [width]*hidden_layer_num

        self.input_layer = nn.Linear(input, width[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_layer_num - 1):
            self.hidden_layers.append(nn.Linear(width[i], width[i+1]))

        self.output_layer = nn.Linear(width[-1], output)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x
    


class AxisScalar2D(nn.Module):
    def __init__(self,net:nn.Module,A:torch.Tensor,B:torch.Tensor) -> None:
        '''X_out=A*X+B'''
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = net
        self.A=A.to(self.device)
        self.B=B.to(self.device)
    
    def forward(self,xy):
        xy_normed = self.A[:] * xy[...,:] + self.B[:]
        return self.net(xy_normed)    

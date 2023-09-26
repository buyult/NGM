import numpy as np
import torch.nn as nn
import torch
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
torch.manual_seed(4)
import os
def multsum(x0):
    sum=0
    x1=np.zeros(len(x0))
    for i in range(len(x0)):
        sum = x0[i]+sum
        x1[i] = sum
    return x1
def multisub(x1):
    b = np.zeros(len(x1))
    b[0] = x1[0]
    for i in range(1,len(x1)):
        b[i]= x1[i]-x1[i-1]
    return b
#MAPE
def err(data, pre_x0, x0):
    sam_nums = len(x0)
    total_err1 = np.sum(np.abs(data[1:sam_nums]-pre_x0[1:sam_nums])/data[1:sam_nums])
    print("MAPE in sample is ", total_err1 / (sam_nums-1))
    total_err2 = np.sum(np.abs(data[sam_nums:] - pre_x0[sam_nums:]) / data[sam_nums:])
    print("MAPE out of sample is ", total_err2 / (len(data)-sam_nums))
    return total_err1 / (sam_nums-1),total_err2 / (len(data)-sam_nums)
class Cat(nn.Module):
    def __init__(self,input_size = 2, hidden_size = 64):
        super(Cat, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
    def forward(self,t, x):
        tt = torch.ones_like(x[:, :, :])*t
        ttx = torch.cat([tt,x],2)
        #print(ttx)
        out = self.rnn(ttx)
        return out
class Linear(nn.Module):
    def __init__(self,hidden_size, output_size):
        super(Linear, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.tanh = nn.ReLU()
    def forward(self, x):
        x = x.view(1, self.hidden_size)
        out = self.linear1(x)
        # out = self.tanh(out)
        out = self.linear2(out)
        out = out.unsqueeze(dim=0)
        #print("dense:",out.shape)
        return out
class ODEfunc(nn.Module):
    def __init__(self, hidden_size = 64, output_size = 1):
        super(ODEfunc, self).__init__()
        self.lstm = Cat(2,hidden_size)
        self.dense = Linear(hidden_size, output_size)
    def forward(self, t, x):
        out, (ht, ct) = self.lstm(t, x)
        #print(out.shape, ht.shape, ct.shape)
        out = self.dense(out)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = 1e-5
    def forward(self, x, tsteps):
        integration_time = torch.Tensor(tsteps).float()
        integration_time = integration_time.type_as(x)
        out = odeint(self.odefunc, x, integration_time, rtol=self.tol, atol=self.tol, method='euler')
        return out

def plot_pre(pred_x,true_x):
    plt.figure()
    datasize = np.arange(0,len(pred_x))
    plt.scatter(datasize, pred_x, marker='x', color='red',s=40,label='pred_x0')
    plt.scatter(datasize, true_x, marker='o', color='black',s=40,label='true_x0')
    plt.legend(loc='best')
    plt.show()
def train(model, iteration, samp_nums, x1, u0, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    end = samp_nums
    time_steps = np.arange(0, end)
    true_x1 = torch.Tensor(x1[0:end])
    for iter in range(iteration):
        out = model(u0, time_steps)
        out = torch.squeeze(out)
        classify_loss = criterion(out, true_x1)
        loss = classify_loss
        if loss <0.0003:
            return
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print("iter:", iter, " loss:", loss.item())
    #pre_x1 = model(u0, time_steps)
    #pre_x1 = torch.squeeze(pre_x1).detach().numpy()
    #print(pre_x1,x1[0:end])
    #plot_pre(pre_x1,x1[0:end])


if __name__ == '__main__':
    #data
    x0 = np.array([0.32587,0.35735,0.40703,0.46562,0.54914,0.61056,0.67911,0.71666, 0.75836, 0.82847, 0.92694, 0.95393, 0.9374, 0.94487])
    test = np.array([0.93869, 0.94592, 0.96989, 0.98561, 1.01189, 1.00517, 1.04507])
    data = np.append(x0,test,axis=0)
    x1 = multsum(data)
    u0 = torch.FloatTensor([[[0.32587]]])
    print(u0.shape)
    datasize = len(data)
    model = torch.load('./model1.pth')
    # model = ODEBlock(ODEfunc())
    # train(model, 10000, 14, x1, u0)
    #predict
    tsteps = np.arange(0, datasize)
    out = model(u0, tsteps)
    pre_x1 = torch.squeeze(out).detach().numpy()
    pre_x0 = np.round(multisub(pre_x1),5)
    err(data, pre_x0, x0)
    print('raw_data：', data)
    print('pre_x0：', pre_x0)
    # plot_pre(pre_x0,data)





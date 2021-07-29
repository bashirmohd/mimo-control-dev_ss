# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Net::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=2, out_channels=24, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/Sequential[cnn_layers]/Conv2d[0]/input.2
        self.module_3 = py_nndct.nn.Module('aten::celu') #Net::Net/Sequential[cnn_layers]/CELU[2]/input.4
        self.module_4 = py_nndct.nn.Conv2d(in_channels=24, out_channels=38, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/Sequential[cnn_layers]/Conv2d[3]/input.5
        self.module_6 = py_nndct.nn.Module('aten::celu') #Net::Net/Sequential[cnn_layers]/CELU[5]/input.7
        self.module_7 = py_nndct.nn.Conv2d(in_channels=38, out_channels=66, kernel_size=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/Sequential[cnn_layers]/Conv2d[6]/input.8
        self.module_9 = py_nndct.nn.Module('aten::celu') #Net::Net/Sequential[cnn_layers]/CELU[8]/87
        self.module_10 = py_nndct.nn.Module('shape') #Net::Net/89
        self.module_11 = py_nndct.nn.Module('reshape') #Net::Net/input.10
        self.module_12 = py_nndct.nn.Linear(in_features=264, out_features=60, bias=True) #Net::Net/Sequential[linear_layers]/Linear[0]/input.11
        self.module_13 = py_nndct.nn.Module('batch_norm_1d',num_features=60, eps=0.0, momentum=0.1) #Net::Net/Sequential[linear_layers]/BatchNorm1d[1]/input.12
        self.module_14 = py_nndct.nn.Module('aten::celu') #Net::Net/Sequential[linear_layers]/CELU[2]/input.13
        self.module_15 = py_nndct.nn.Linear(in_features=60, out_features=8, bias=True) #Net::Net/Sequential[linear_layers]/Linear[3]/input.14
        self.module_16 = py_nndct.nn.Module('batch_norm_1d',num_features=8, eps=0.0, momentum=0.1) #Net::Net/Sequential[linear_layers]/BatchNorm1d[4]/input.15
        self.module_17 = py_nndct.nn.Module('aten::celu') #Net::Net/Sequential[linear_layers]/CELU[5]/input
        self.module_18 = py_nndct.nn.Module('transpose') #Net::Net/Sequential[linear_layers]/Linear[6]/115
        self.module_19 = py_nndct.nn.Module('matmul') #Net::Net/Sequential[linear_layers]/Linear[6]/116
        self.linear_layers_6_weight = torch.nn.parameter.Parameter(torch.Tensor(8, 8))

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_3 = self.module_3(alpha=1.0, input=self.output_module_1)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_6 = self.module_6(alpha=1.0, input=self.output_module_4)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_9 = self.module_9(alpha=1.0, input=self.output_module_7)
        self.output_module_10 = self.module_10(input=self.output_module_9, dim=0)
        self.output_module_11 = self.module_11(input=self.output_module_9, size=[self.output_module_10,-1])
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_13 = self.module_13(self.output_module_12)
        self.output_module_14 = self.module_14(alpha=1.0, input=self.output_module_13)
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_16 = self.module_16(self.output_module_15)
        self.output_module_17 = self.module_17(alpha=1.0, input=self.output_module_16)
        self.output_module_18 = self.module_18(dim1=1, dim0=0, input=self.linear_layers_6_weight)
        self.output_module_19 = self.module_19(other=self.output_module_18, input=self.output_module_17)
        return self.output_module_19

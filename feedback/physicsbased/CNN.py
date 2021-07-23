from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm1d, BatchNorm2d, AvgPool2d, SELU, ELU, CELU

class Net(Module):

    def __init__(self, double_frame, cnn_input):
        super(Net, self).__init__()

        if double_frame: # Input changes from 5*5 to 10*5
            self.cnn_layers = Sequential(
                # Defining 1st 2D convolution layer, Conv2d( in_channels, out_channels, kernel_size, stride=1, padding=0)
                Conv2d( 2, 24, kernel_size=(2,2), stride=(1,1), padding=(0,0) ),
                BatchNorm2d(24),
                # ReLU(inplace=True),
                CELU(alpha=1.0, inplace=False),
                # MaxPool2d(kernel_size=(3,2), stride=(1,1)),
                # AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0,0) ),

                # Defining 2nd 2D convolution layer
                Conv2d(24, 38, kernel_size=(2,2), stride=(1,1), padding=(0,0) ),
                BatchNorm2d(38),
                # ReLU(inplace=True),
                CELU(alpha=1.0, inplace=False),

                # # # Defining 3rd 2D convolution layer
                Conv2d(38, 66, kernel_size=(2, 2), stride=(1, 1), padding=(0,0) ),
                BatchNorm2d(66),
                # ReLU(inplace=True),
                CELU(alpha=1.0, inplace=False),
                #
                # # # Defining 4th 2D convolution layer
                # Conv2d(48, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0,0) ),
                # BatchNorm2d(48),
                # # ReLU(inplace=True),
                # CELU(alpha=1.0, inplace=False),
                #
                # # # Defining 5th 2D convolution layer
                # Conv2d(48, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
                # BatchNorm2d(48),
                # # ReLU(inplace=True),
                # CELU(alpha=1.0, inplace=False),
                #
                # # # Defining 6th 2D convolution layer
                # Conv2d(48, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)),
                # BatchNorm2d(48),
                # # ReLU(inplace=True),
                # CELU(alpha=1.0, inplace=False),

            )

            self.linear_layers = Sequential(
                # 1st dense layer
                Linear(66 * 2 * 2, 60),
                BatchNorm1d(60),
                # ReLU(inplace=True),
                CELU(alpha=1.0, inplace=False),

                # 2st dense layer
                Linear(60, 8),
                BatchNorm1d(8),
                # ReLU(inplace=True),
                CELU(alpha=1.0, inplace=False),

                Linear(8, 8, bias=False),
            )
        else:
            self.cnn_layers = Sequential(
                # Defining 1st 2D convolution layer, Conv2d( in_channels, out_channels, kernel_size, stride=1, padding=0)
                Conv2d(1, 24, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
                BatchNorm2d(24),
                # SELU(inplace=False),
                CELU(alpha=1.0, inplace=False),
                # MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
                # ELU(alpha=1.0, inplace=False),
                # ReLU(inplace=False),
                # AvgPool2d(kernel_size=(1, 3), stride=(1, 2)),

                # Defining 2nd 2D convolution layer
                Conv2d(24, 38, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
                BatchNorm2d(38),
                # ReLU(inplace=False),
                CELU(alpha=1.0, inplace=False),
                # ReLU(inplace=False),
                # ELU(alpha=1.0, inplace=False),
                # SELU(inplace=False),

                # Defining 3rd 2D convolution layer
                Conv2d(38, 66, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
                BatchNorm2d(66),
                # ReLU(inplace=False),
                CELU(alpha=1.0, inplace=False),
                # ReLU(inplace=False),
                # ELU(alpha=1.0, inplace=False),
                # SELU(inplace=False),

                # # # Defining 4th 2D convolution layer
                # Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=0),
                # BatchNorm2d(32),
                # # ReLU(inplace=False),
                # CELU(alpha=1.0, inplace=False),
                # # ReLU(inplace=False),
                # # ELU(alpha=1.0, inplace=False),
                # # SELU(inplace=False),
                #
                # # # # Defining 5th 2D convolution layer
                # Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=0),
                # BatchNorm2d(32),
                # CELU(alpha=1.0, inplace=False),
                # # ReLU(inplace=False),

            )

            self.linear_layers = Sequential(
                # 1st dense layer
                Linear(66 * 2 * 2, 60),
                BatchNorm1d(60),
                # ReLU(inplace=False),
                CELU(alpha=1.0, inplace=False),
                # ELU(alpha=1.0, inplace=False),
                # SELU(inplace=False),

                # 2st dense layer
                Linear(60, 8),
                BatchNorm1d(8),
                # ReLU(inplace=False),
                CELU(alpha=1.0, inplace=False),
                # ELU(alpha=1.0, inplace=False),
                # SELU(inplace=False),

                Linear(8, 8, bias=False),
            )
    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
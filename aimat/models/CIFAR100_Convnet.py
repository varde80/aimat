from torch import nn
import torch.nn.functional as F

class CIFAR100_ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()        

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

class MNIST_ConvNet(nn.Module):
    """Generalized CNN model for image dataset

    Attributes:
        - convs (torch.nn.modules.container.ModuleList):   List with the convolutional layers
        - conv2_drop (torch.nn.modules.dropout.Dropout2d): Dropout for conv layer 2
        - in_size (int):                                   Size of input image
        - out_feature (int):                               Size of flattened features
        - fc1 (torch.nn.modules.linear.Linear):            Fully Connected layer 1
        - fc2 (torch.nn.modules.linear.Linear):            Fully Connected layer 2
        - p1 (float):                                      Dropout ratio for FC1

    Methods:
        - forward(x):
    """
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1):
        """Parameters:
            - trial (optuna.trial._trial.Trial): Optuna trial
            - num_conv_layers (int):             Number of convolutional layers
            - num_filters (list):                Number of filters of conv layers
            - num_neurons (int):                 Number of neurons of FC layers
            - drop_conv2 (float):                Dropout ratio for conv layer 2
            - drop_fc1 (float):                  Dropout ratio for FC1
        """
        super(MNIST_ConvNet, self).__init__()                                                     # Initialize parent class
        in_size = 28                                                                    # Input image size (28 pixels)
        kernel_size = 3                                                                 # Convolution filter size

        # Define the convolutional layers
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], kernel_size=(3, 3))])  # List with the Conv layers
        out_size = in_size - kernel_size + 1                                            # Size of the output kernel
        out_size = int(out_size / 2)                                                    # Size after pooling
        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=(3, 3)))
            out_size = out_size - kernel_size + 1                                       # Size of the output kernel
            out_size = int(out_size/2)                                                  # Size after pooling

        self.conv2_drop = nn.Dropout2d(p=drop_conv2)                                    # Dropout for conv2
        self.out_feature = num_filters[num_conv_layers-1] * out_size * out_size         # Size of flattened features
        self.fc1 = nn.Linear(self.out_feature, num_neurons)                             # Fully Connected layer 1
        self.fc2 = nn.Linear(num_neurons, 10)                                           # Fully Connected layer 2
        self.p1 = drop_fc1                                                              # Dropout ratio for FC1

        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')

    def forward(self, x):
        """Forward propagation.

        Parameters:
            - x (torch.Tensor): Input tensor of size [N,1,28,28]
        Returns:
            - (torch.Tensor): The output tensor after forward propagation [N,10]
        """
        for i, conv_i in enumerate(self.convs):  # For each convolutional layer
            if i == 2:  # Add dropout if layer 2
                x = F.relu(F.max_pool2d(self.conv2_drop(conv_i(x)), 2))  # Conv_i, dropout, max-pooling, RelU
            else:
                x = F.relu(F.max_pool2d(conv_i(x), 2))                   # Conv_i, max-pooling, RelU

        x = x.view(-1, self.out_feature)                     # Flatten tensor
        x = F.relu(self.fc1(x))                              # FC1, RelU
        x = F.dropout(x, p=self.p1, training=self.training)  # Apply dropout after FC1 only when training
        x = self.fc2(x)                                      # FC2

        return F.log_softmax(x, dim=1)                       # log(softmax(x))


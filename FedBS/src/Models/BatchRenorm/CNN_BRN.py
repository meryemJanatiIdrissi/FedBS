from torch import nn
import torch.nn.functional as F
from . import BatchRenorm as brn



class CNNCifar_BRN(nn.Module):

    def __init__(self, conv1_dim=100, conv2_dim=150, conv3_dim=250, conv4_dim=500):
        super(CNNCifar_BRN, self).__init__()
        self.conv4_dim = conv4_dim

        self.conv1 = nn.Conv2d(3, conv1_dim, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(conv1_dim, conv2_dim, 3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(conv2_dim, conv3_dim, 3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(conv3_dim, conv4_dim, 3, stride=1, padding=2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(conv4_dim * 3 * 3, 270) # 3x3 is precalculated and written, you need to do it if you want to change the # of filters
        self.fc2 = nn.Linear(270, 150)
        self.fc3 = nn.Linear(150, 10)

        self.normalize1 = brn.BatchRenorm2d(conv1_dim) #nn.BatchNorm2d(conv1_dim) #   #mbn.My
        self.normalize2 = brn.BatchRenorm2d(conv2_dim) #nn.BatchNorm2d(conv2_dim) #  #
        self.normalize3 = brn.BatchRenorm2d(conv3_dim) #  nn.BatchNorm2d(conv3_dim) #
        self.normalize4 = brn.BatchRenorm2d(conv4_dim) #   nn.BatchNorm2d(conv4_dim) #

    def forward(self, x):
        x = self.pool(F.relu(self.normalize1((self.conv1(x))))) # first convolutional then batch normalization then relu then max pool
        x = self.pool(F.relu(self.normalize2((self.conv2(x)))))
        x = self.pool(F.relu(self.normalize3((self.conv3(x)))))
        x = self.pool(F.relu(self.normalize4((self.conv4(x)))))

        x = x.view(-1, self.conv4_dim * 3 * 3) # flattening the features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class CNNCifar_BRN_1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.BNorm1 = brn.BatchRenorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3)
        self.BNorm2 = brn.BatchRenorm2d(32)
        self.fc1 = nn.Linear(32*6*6,256)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,10)
        self.drop = nn.Dropout(p =0.2)

    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        out = self.BNorm1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        out = self.BNorm2(out)
        out = out.view(out.shape[0],-1)

        out = self.fc1(out)
        out = self.drop(F.relu(out))
        out = self.fc2(out)
        out = self.drop(F.relu(out))
        final = F.log_softmax(F.relu(self.fc3(out)) , dim = 1)

        return final

class CNNMnist_BRN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNMnist_BRN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            brn.BatchRenorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            brn.BatchRenorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNFMnist_BRN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNFMnist_BRN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            brn.BatchRenorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            brn.BatchRenorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


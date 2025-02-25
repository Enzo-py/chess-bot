import torch
from torch import nn
from torch.nn import functional as F

class Embedding(nn.Module):

    def __init__(self):
        super(Embedding, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1) # [b, 8, 8, 14] -> [b, 8, 8, 64]
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8) # [b, 8, 8 14] -> [b, 1, 1, 64]

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        x: [batch_size, 8, 8, 14]
        """

        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv0(x))
        x = self.dropout(x)
        x = F.relu(self.conv1(x))

        x = x.view(x.shape[0], -1)
        return x
    
class DefaultClassifier(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        """
        x: [batch_size, 128]
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
class DefaultDecoder(nn.Module):
    """from latent to board"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_board = nn.Linear(1024, 8*8*12)
        
        self.fc_turn = nn.Linear(1024, 1)

    def forward(self, x):
        """
        x: [batch_size, 256]
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_board = self.fc_board(x)
        x_board = x_board.view(-1, 8, 8, 12)

        x_turn = self.fc_turn(x)
        return x_board, x_turn
    
class DefaultScoreHead(nn.Module):
    """give the value of the pieces"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2) # [black_score, white_score]

    def forward(self, x):
        """
        x: [batch_size, 256]
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
  
class DefaultGenerativeHead(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(512, 128)
        self.move = nn.Linear(128, 64*64*5)

    def forward(self, x):
        """
        x: [batch_size, 256]
        """
        x = F.relu(self.fc1(x))
        move = self.move(x)
        return move

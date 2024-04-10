import torch
import torch.nn as nn
import torch.nn.functional as F



######################################################## Model for Assignment 10 ########################################################

#Define the model
class Model_CustomResNet(nn.Module):
    def __init__(self,dropout_value = 0.01):
        super(Model_CustomResNet, self).__init__()


        # Specify normalization technique
        self.xnorm = lambda inp: nn.BatchNorm2d(inp)




        ##################### PREP LAYER STARTS ################

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            self.xnorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )  #

        ##################### PREP LAYER ENDS ################


        ################ LAYER 1 STARTS #################


        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            self.xnorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )  #

        ###### RESNET BLOCK
        self.layer1_resblock = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            self.xnorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            self.xnorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) #


        ################ LAYER 1 ENDS #################
        


        ################ LAYER 2 STARTS #################


        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            self.xnorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )  # 

        ################ LAYER 2 ENDS #################




        ################ LAYER 3 STARTS #################


        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            self.xnorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        ###### RESNET BLOCK
        self.layer3_resblock = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            self.xnorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            self.xnorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )


        ################ LAYER 3 ENDS #################


        # Max pool operation
        self.pool = nn.MaxPool2d(4, 4)


        ####### FC LAYER
        self.fc = nn.Linear(512,10)



    def forward(self, x):

        # Prep Layer
        x = self.preplayer(x)

        # Layer1
        x = self.layer1(x)
        x = x + self.layer1_resblock(x)

        
        # Layer2
        x = self.layer2(x)

        # Layer3
        x = self.layer3(x)
        x = x +  self.layer3_resblock(x)
          

        # pooling
        x = self.pool(x)

        x = x.view(x.size(0),-1)        # Flatten for softmax

        # fc layer
        x = self.fc(x)
        
        return F.log_softmax(x,dim=1)   # Apply log_softmax activation function on final output values





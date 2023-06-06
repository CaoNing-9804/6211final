from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        
        #batchsize*26*96
        #transpose: 

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=8,
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(96, 192, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        #batch*192*8?

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        self.conv_block5 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        self.conv_block6 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = x_in.transpose(2,1)
        #print(x.shape)
        x = self.conv_block1(x)
        #print(x.shape)
        x = self.conv_block2(x)
        #print(x.shape)
        x = x.transpose(2,1)
        x = self.conv_block4(x)
        #print(x.shape)
        x = self.conv_block5(x)
        #print(x.shape)
        x = self.conv_block6(x)
        #print(x.shape)
        x = self.conv_block7(x)
        #print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

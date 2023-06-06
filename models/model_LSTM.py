from torch import nn
import torch
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        
        #batchsize*26*96 
        self.input_channels=configs.input_channels
        self.final_out_channels=configs.final_out_channels
        self.features_len=configs.features_len
        self.bilstm = nn.LSTM(input_size = configs.input_channels, 
                              hidden_size = configs.final_out_channels, 
                              num_layers = 2, 
                              batch_first = True, 
                              dropout = configs.dropout, 
                              bidirectional = True)
        self.combine=nn.Linear(2*configs.final_out_channels,configs.final_out_channels)
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)
        
    def forward(self, x_in):
        #in_150*configs.input_channels*96
        #out 150*configs.final_out_channels*14 150*3
        #print(x_in.shape)
        x_in=x_in.reshape(x_in.shape[0],96,self.input_channels)
        #print(x_in.shape)
        
        h = torch.zeros(4,x_in.shape[0], self.final_out_channels).to("cuda")
        c = torch.zeros(4,x_in.shape[0], self.final_out_channels).to("cuda")
        #x = x_in.transpose(2,1)
        #print(x_in.shape,h.shape)
        x, _ = self.bilstm(x_in,(h,c))
        #print(x.shape)
        # 150,configs.features_len,configs.final_out_channels*2
        
        x=self.combine(x)
        #print(x.shape)
        # 150,configs.features_len,configs.final_out_channels
        x_flat = x.reshape(x.shape[0], -1)
        x_flat=x_flat[:,-self.features_len*self.final_out_channels:]
        #print(x_flat.shape)
        # 150,configs.features_len*configs.final_out_channels
        logits = self.logits(x_flat)
        return logits, x
from torch import nn
import torch
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        self.input_channels=configs.input_channels
        self.final_out_channels=configs.final_out_channels
        self.features_len=configs.features_len
        self.transformerLayer=nn.TransformerEncoderLayer(
                              d_model=configs.input_channels, 
                              nhead=2, 
                              dim_feedforward=configs.final_out_channels,
                              dropout=configs.dropout, 
                              layer_norm_eps=1e-05, 
                              batch_first=True, 
                              device="cuda")
        self.Transformerencoder=nn.TransformerEncoder(self.transformerLayer,num_layers=5)
        self.logits = nn.Linear(configs.features_len * configs.final_out_channels, configs.num_classes)
        
    def forward(self, x_in):
        #in_150*configs.input_channels*96
        #out 150*configs.final_out_channels*14 150*3
        #print(x_in.shape)
        x_in=x_in.reshape(x_in.shape[0],96,self.input_channels)
        #print(x_in.shape)
        
        x=self.Transformerencoder(x_in)
        #print(x.shape)
        # 150,configs.features_len,configs.final_out_channels
        
        #print(x.shape)
        # 150,configs.features_len,configs.final_out_channels
        x_flat = x.reshape(x.shape[0], -1)
        x_flat=x_flat[:,-self.features_len*self.final_out_channels:]
        #print(x_flat.shape)
        # 150,configs.features_len*configs.final_out_channels
        logits = self.logits(x_flat)
        return logits, x
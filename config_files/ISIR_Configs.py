class Config_random_init(object):
    def __init__(self):
        # model configs
        self.input_channels = 26
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 3
        self.dropout = 0.8
        self.features_len = 14

        # training configs
        self.num_epoch = 5

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.5

        # data parameters
        self.drop_last = False
        self.batch_size = 256

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.basemdl = "LSTM"



class Config_self_supervised(object):
    def __init__(self):
        # model configs
        self.input_channels = 26
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 3
        self.dropout = 0.15
        self.features_len = 14

        # training configs
        self.num_epoch = 50

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr =1

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.basemdl = "TS"
        
        
        
class Config_train_linear(object):
    def __init__(self):
        # model configs
        self.input_channels = 26
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 3
        self.dropout = 0.15
        self.features_len = 14

        # training configs
        self.num_epoch = 10

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-2

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.basemdl = "TS"
        



class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.2
        self.jitter_ratio = 5
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 3

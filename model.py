import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.onnx as torch_onnx
from utils import *
import operator
import Configuration

cfg = Configuration.Config()

def generate_model(pretrained_weights=cfg.pretrained_weights):
    model = UNet_ConvLSTM(cfg.input_image_c, cfg.n_class)
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))
    return model


class UNet_ConvLSTM(nn.Module):

    def __init__(self, n_channels, n_classes):

        super(UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, cfg.encoder_filters_num_1, cfg.encoder_droprate)
        self.down1 = down(cfg.encoder_filters_num_1, cfg.encoder_filters_num_2, cfg.encoder_droprate)
        self.down2 = down(cfg.encoder_filters_num_2, cfg.encoder_filters_num_3, cfg.encoder_droprate)
        self.down3 = down(cfg.encoder_filters_num_3, cfg.encoder_filters_num_4, cfg.encoder_droprate)
        self.down4 = down(cfg.encoder_filters_num_4, cfg.encoder_filters_num_4, cfg.encoder_droprate)
        self.up1 = up(cfg.decoder_filters_num_1*2, cfg.decoder_filters_num_2, cfg.decoder_droprate)
        self.up2 = up(cfg.decoder_filters_num_2*2, cfg.decoder_filters_num_3, cfg.decoder_droprate)
        self.up3 = up(cfg.decoder_filters_num_3*2, cfg.decoder_filters_num_4, cfg.decoder_droprate)
        self.up4 = up(cfg.decoder_filters_num_4*2, cfg.decoder_filters_num_4, cfg.decoder_droprate)
        self.outc = outconv(cfg.decoder_filters_num_4, n_classes)
        self.convlstm = ConvLSTM(input_size=cfg.rnn_shape,
                                 input_dim=cfg.encoder_filters_num_4,
                                 hidden_dim=cfg.rnn_hiden_filters,
                                 kernel_size=cfg.rnn_kernel_size,
                                 num_layers=cfg.rnn_num_layers,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False,
                                 drop_rate=cfg.rnn_dropout)

    def forward(self, x):

        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][ -1,:, :, :, :]
        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x #, test


if __name__ == '__main__':
    model = generate_model(None)
    # model.eval()
    # example = torch.rand((1, 5, 3, 256, 512)).cuda()
    # output = torch_onnx.export(model.cuda(), example, "model.onnx")
    # traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save("model.pt")
    print("Complite")
    

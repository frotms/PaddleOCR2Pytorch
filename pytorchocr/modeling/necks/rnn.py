import os, sys
import torch
import torch.nn as nn

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        # x = x.transpose([0, 2, 1])  # paddle (NTC)(batch, width, channels)
        x = x.permute(0,2,1)
        return x


class EncoderWithRNN_(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn1 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
        self.rnn2 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True) # batch_first:=True

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)
        return x
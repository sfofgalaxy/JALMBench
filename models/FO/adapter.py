import torch

from torch import nn

class CNNAdapter(torch.nn.Module):
    def __init__(
        self,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 5,
    ):
        super().__init__()
        
        self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
        
        self.conv1d1 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 1, 0)
        self.conv1d2 = nn.Conv1d(2 * enc_out_dim, 4 * enc_out_dim, kernel_size, 1, 0)
        
        self.bn1 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
        self.bn2 = nn.BatchNorm1d(4 * enc_out_dim, eps=1e-3, momentum=0.99)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.project = nn.Linear(4 * enc_out_dim, llm_embed_dim)
    
    def forward(self, x, mask_pad):
        """
            x: B, T, enc_out_dim
            mask: (B, T) or (B, 1, T)
        """
        x = x.transpose(1, 2)  # B, channels, T

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        x = self.left_padding1(x)
        x = self.conv1d1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.left_padding2(x)
        x = self.conv1d2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        return x, mask_pad

class LinearAdapter(torch.nn.Module):
    def __init__(
        self,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
    ):
        super().__init__()

        self.adpter = torch.nn.Linear(enc_out_dim, llm_embed_dim)

    def forward(self, x, mask_pad):
        return self.adpter(x), mask_pad

class CNNSubsampling(torch.nn.Module):
    def __init__(
        self,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 5,
        activation_func: str = 'relu',
        norm: str = 'batch',
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        if enc_out_dim * 4 < llm_embed_dim:
            self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d1 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 1, 0)
            self.bn1 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
            self.relu1 = nn.ReLU()

            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(2 * enc_out_dim, 4 * enc_out_dim, kernel_size, 2, 0)
            self.bn2 = nn.BatchNorm1d(4 * enc_out_dim, eps=1e-3, momentum=0.99)
            self.relu2 = nn.ReLU()
            
            self.project = nn.Linear(4 * enc_out_dim, llm_embed_dim)
            self.cnn_num = 2
        else:
            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 2, 0)
            if norm == 'batch':
                self.bn2 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
            elif norm == 'layer':
                self.bn2 = nn.LayerNorm(2 * enc_out_dim, eps=1e-3)
            if activation_func == 'gelu':
                self.relu2 = nn.GELU()
            else:
                self.relu2 = nn.ReLU()
            
            self.project = nn.Linear(2 * enc_out_dim, llm_embed_dim)
            self.cnn_num = 1
    
    def forward(self, x, mask_pad, cache=None, return_cache=False):
        """
            x: B, T, enc_out_dim
            mask: (B, T) or (B, 1, T)
        """
        x = x.transpose(1, 2)  # B, channels, T

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.cnn_num == 2:
            if cache is None:
                x = self.left_padding1(x)
            else:
                x = torch.cat((cache[1], x), dim=2)
            if cache is not None:
                cache[1] = x[:, :, 1-self.kernel_size:]
            else:
                cache = [None, x[:, :, 1-self.kernel_size:]]
            x = self.conv1d1(x)
            x = self.bn1(x)
            x = self.relu1(x)

        if cache is None or cache[0] is None:
            x = self.left_padding2(x)
        else:
            x = torch.cat((cache[0], x), dim=2)
        if cache is not None:
            cache[0] = x[:, :, 1-self.kernel_size:]
        else:
            cache = [x[:, :, 1-self.kernel_size:]]
        x = self.conv1d2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.bn2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.relu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        if return_cache:
            return x, mask_pad[:, :, 0::2], cache
        return x, mask_pad[:, :, 0::2]

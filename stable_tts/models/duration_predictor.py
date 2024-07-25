import torch
import torch.nn as nn

# modified from https://github.com/jaywalnut310/vits/blob/main/models.py#L98
class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = nn.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = nn.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g):
    x = x.detach()
    x = x + self.cond(g.unsqueeze(2).detach())
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x.transpose(1,2)).transpose(1,2)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x.transpose(1,2)).transpose(1,2)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask
  
def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss
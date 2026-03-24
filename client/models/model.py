"""
Shape2Gesture モデル定義
- HandVAE: 手形状生成VAE (23関節 × 3次元)
- PartsEncoder_w_TNet: 接触部位特徴エンコーダ (T-Net付き)
- Position_Generater_VAE: 手首位置・回転角生成VAE
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class VAE_Encoder(nn.Module):
    def __init__(self):
        super(VAE_Encoder, self).__init__()
        self.d1 = nn.Linear(66, 66)
        self.d2 = nn.Linear(66, 66)
        self.d3 = nn.Linear(66, 66)
        self.d4 = nn.Linear(66, 64)
        self.d5 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(66)
        self.bn2 = nn.BatchNorm1d(66)
        self.bn3 = nn.BatchNorm1d(66)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)

    def forward(self, x):
        fc1 = F.relu(self.bn1(self.d1(x)))
        fc2 = F.relu(self.bn2(self.d2(fc1)))
        fc3 = F.relu(self.bn3(self.d3(fc2)))
        fc4 = F.relu(self.bn4(self.d4(fc3)))
        fc5 = F.relu(self.bn5(self.d5(fc4)))
        return fc5


class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder, self).__init__()
        self.up1 = nn.Linear(16, 32)
        self.up2 = nn.Linear(32, 64)
        self.up3 = nn.Linear(64, 66)
        self.up4 = nn.Linear(66, 66)
        self.up5 = nn.Linear(66, 66)
        self.up6 = nn.Linear(66, 66)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(66)
        self.bn4 = nn.BatchNorm1d(66)
        self.bn5 = nn.BatchNorm1d(66)

    def forward(self, x):
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        x = F.relu(self.bn3(self.up3(x)))
        x = F.relu(self.bn4(self.up4(x)))
        x = F.relu(self.bn5(self.up5(x)))
        x = F.sigmoid(self.up6(x))
        return x


class HandVAE(nn.Module):
    """
    手形状生成VAE
    入力: 16次元潜在ベクトル
    出力: 23関節 × 3次元 = 66次元の手形状
    """
    def __init__(self):
        super(HandVAE, self).__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.fc_mu = nn.Linear(32, 16)
        self.fc_logvar = nn.Linear(32, 16)

    def encode(self, x):
        fc5 = self.encoder(x)
        mu = self.fc_mu(fc5)
        logvar = self.fc_logvar(fc5)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = torch.rand_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z)
        return pred, mu, logvar

    def finetune(self, x):
        """推論時: 16次元特徴から手形状を直接デコード"""
        pred = self.decoder(x)
        return pred


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat_new(nn.Module):
    def __init__(self):
        super(PointNetfeat_new, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x, isgesture=True):
        if isgesture:
            x = self.stn(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class PartsEncoder_w_TNet(torch.nn.Module):
    """
    接触部位点群から16次元潜在ベクトルを生成するエンコーダ
    入力: parts点群(B,3,256) + 全体形状特徴(B,1024)
    出力: (z, mu, logvar) それぞれ16次元
    """
    def __init__(self, feature_transform=False):
        super(PartsEncoder_w_TNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)
        self.fc_mu = nn.Linear(32, 16)
        self.fc_logvar = nn.Linear(32, 16)

    def reparameterize(self, mu, logvar):
        epsilon = torch.rand_like(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + epsilon * std
        return z

    def forward(self, parts, all_feat):
        x = self.feat(parts, True)
        x = torch.cat([x, all_feat], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        x = self.reparameterize(mu, logvar)
        return x, mu, logvar


class PartsEncoder_wo_TNet(torch.nn.Module):
    """Position_Generater_VAE内部で使用するエンコーダ (T-Netなし)"""
    def __init__(self, feature_transform=False):
        super(PartsEncoder_wo_TNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat_new()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)

    def forward(self, parts, all_feat):
        x = self.feat(parts, False)
        x = torch.cat([x, all_feat], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        return x


class Quaterion_Decodernew(nn.Module):
    """Z軸周りの回転角 (cosθ, sinθ) を生成するデコーダ"""
    def __init__(self):
        super(Quaterion_Decodernew, self).__init__()
        self.fn1 = nn.Linear(16, 32)
        self.fn2 = nn.Linear(32, 32)
        self.fn3 = nn.Linear(32, 32)
        self.fn4 = nn.Linear(32, 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fn1(x)))
        x = F.tanh(self.bn2(self.fn2(x)))
        x = F.tanh(self.bn3(self.fn3(x)))
        x = F.tanh(self.bn4(self.fn4(x)))
        x = F.normalize(x, dim=-1)
        return x


class WristPosition_Decoder(nn.Module):
    """手首位置 (x, y, z) を生成するデコーダ"""
    def __init__(self):
        super(WristPosition_Decoder, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.bn3(self.fc3(x))
        x = 2 * F.tanh(self.fc4(x))
        return x


class Position_Generater_VAE(nn.Module):
    """
    手首位置・回転角を同時に生成するVAE
    入力: parts点群 + 全体形状特徴
    出力: (回転角ベクトル, 手首位置, KLDロス, z)
    """
    def __init__(self):
        super(Position_Generater_VAE, self).__init__()
        self.encoder = PartsEncoder_wo_TNet()
        self.decoder = Quaterion_Decodernew()
        self.fc_mu1 = nn.Linear(32, 16)
        self.fc_logvar1 = nn.Linear(32, 16)
        self.decode_wrist = WristPosition_Decoder()

    def encode(self, x, allfeat):
        x = self.encoder(x, allfeat)
        mu_r = self.fc_mu1(x)
        logvar_r = self.fc_logvar1(x)
        return mu_r, logvar_r

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_multi(self, mu, logvar, N):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mu.size(0), N, mu.size(1), device=mu.device)
        return mu.unsqueeze(1) + eps * std.unsqueeze(1)

    def forward(self, x, allfeat, N):
        mu, logvar = self.encode(x, allfeat)
        z = self.reparameterize_multi(mu, logvar, N)
        B, N, D = z.shape
        z_flat = z.view(B * N, D)
        pred = self.decoder(z_flat).view(B, N, -1)
        pred_wrist = self.decode_wrist(z_flat).view(B, N, -1)
        return pred, pred_wrist, self.kld_loss(mu, logvar), z.detach().cpu().numpy()

    def kld_loss(self, mu, logvar):
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return kld

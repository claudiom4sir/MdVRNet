"""
Definition of the MdVRNet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
from estimate_params import DPEN


class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=True),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)

class ResBlock(nn.Module):
	def __init__(self, ch):
		super(ResBlock, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return x + self.net(x)

class FullResBranch(nn.Module):
	def __init__(self, in_ch, out_ch, int_ch=64):
		super(FullResBranch, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_ch, int_ch, kernel_size=3, padding=1, bias=False),
			ResBlock(int_ch),
			ResBlock(int_ch),
			ResBlock(int_ch),
			nn.Conv2d(int_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.net(x)

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3 + 2), num_in_frames*self.interm_ch, kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class MRB(nn.Module):
	""" Definition of the MRB of MdVRNet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(MRB, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		# preprocessing layer
		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		# low resolution branch
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0 * 2, out_ch=3)
		# full resolution branch
		self.resnet = FullResBranch(in_ch=self.chs_lyr0, out_ch=self.chs_lyr0)
		# attention layer
		self.se = SELayer(channel=self.chs_lyr0 * 2, reduction=8)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))

		res_out = self.resnet(x0)

		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		cat = torch.cat(((x0 + x1), res_out), dim=1)
		x = self.outc(self.se(cat))

		# Residual
		x = in1 - x

		return x

class MdVRNet(nn.Module):
	""" Definition of the MdVRNet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(MdVRNet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each stage
		self.temp1 = MRB(num_input_frames=3)
		self.temp2 = MRB(num_input_frames=3)
		self.estimate_sigma_1 = DPEN() # IGNORE THIS LINE
		self.estimate_sigma_2 = DPEN() # IGNORE THIS LINE
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x
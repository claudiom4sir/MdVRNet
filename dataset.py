"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import random
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
from utils import open_sequence
import cv2
from utils import apply_jpeg_artifacts
import numpy as np
from random import randint


NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))

		# open individual sequences and append them to the sequence list
		sequences = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			sequences.append(seq)

		self.sequences = sequences

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index])

	def __len__(self):
		return len(self.sequences)

class DPENDataset(Dataset):

	def __init__(self, path_data, min_sigma=5, max_sigma=55, min_q=15, max_q=35, patch_size=64, is_test=False):
		self.path_data = path_data
		self.min_sigma = min_sigma
		self.max_q = max_q
		self.min_q = min_q
		self.max_sigma = max_sigma
		self.patch_size = patch_size
		self.is_test = is_test
		image_paths = []
		self.images = []
		for folder in os.listdir(path_data):
			for im in os.listdir(path_data + "/" + folder):
				image_paths.append(path_data + "/" + folder + "/" + im)
		print("Loading images...")
		for im in image_paths:
				im = cv2.imread(im)
				im = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
				self.images.append(im)
		print("Done. Loaded %d images" % len(self.images))

	def mirror_image(self, im):
		return cv2.flip(im, 0)

	def crop_image(self, im, x, y):
		return im[x:x + self.patch_size, y:y + self.patch_size, :]
  
	def flip_image(self, im):
		return cv2.flip(im, 1)

	def augment_data(self, img):
		# data augmentation
		flip = False
		mirror = False
		if random.random() > 0.5:
			flip = True
		if random.random() > 0.5:
			mirror = True
		# crop param
		h = random.randint(0, img.shape[0] - self.patch_size)
		w = random.randint(0, img.shape[1] - self.patch_size)
		img = self.crop_image(img, h, w)
		if flip:
			img = self.flip_image(img)
		if mirror:
			img = self.mirror_image(img)
		return img

	def __getitem__(self, index):
		q = randint(self.min_q, self.max_q)
		sigma = random.uniform(self.min_sigma, self.max_sigma)
		sigma /= 255.
		img = self.images[index]
		if not self.is_test:
			img = self.augment_data(img)
		img = torch.from_numpy(np.float32(img / 255.)).permute(2, 0, 1)
		noise = torch.empty_like(img).normal_(mean=0, std=sigma)
		img += noise
		img = torch.clamp(img, 0.,1.)
		img = apply_jpeg_artifacts(img, q)
		img = torch.clamp(img, 0.,1.)
		q /= 100.
		return img, torch.FloatTensor([sigma]), torch.FloatTensor([q])

	def __len__(self):
		return len(self.images)





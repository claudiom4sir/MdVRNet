from torch.utils.data.dataset import Dataset
import numpy as np
from dataset import DPENDataset
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import torch
import argparse
import random
from estimate_params import DPEN
import torch.nn as nn


def main(**args):
	# set seed to ensure reproducibility
	seed = args['random_seed']
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False # CUDNN optimization
	torch.backends.cudnn.deterministic = True
	
	# load datasets
	valset = DPENDataset(args['valset_dir'], min_sigma=args['noise'], max_sigma=args['noise'], min_q=args['q'], max_q=args['q'], patch_size=args['patch_size'])
	val_dl = DataLoader(valset)

	# create DPEN model and set training params
	model = DPEN()
	model.load_state_dict(torch.load(args['estimate_parameter_model']))
	model = model.cuda()
	criterion = nn.L1Loss(reduction='sum').cuda()



	# evaluate the model
	val_loss = 0
	val_sigma_loss = 0
	val_q_loss = 0
	model.eval()
	with torch.no_grad():
		for _, data in enumerate(val_dl):
			data, sigma, q = data
			data = data.cuda()
			sigma = sigma.cuda().squeeze(1)
			q = q.cuda().squeeze(1)
			out_sigma, out_q = model(data)
			loss = criterion(out_sigma, sigma) + criterion(out_q, q)
			val_loss += loss.item()
			val_sigma_loss += criterion(out_sigma, sigma).item()
			val_q_loss += criterion(out_q, q).item()
	val_loss /= len(valset)
	val_sigma_loss /= len(valset)
	val_q_loss /= len(valset)

	print("Sigma_error:%f, Q_error:%f" % (val_sigma_loss*255., val_q_loss*100))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Test DPEN")
	parser.add_argument("--random_seed", type=int, default=0, help="Random seed to ensure reproducibility")
	parser.add_argument("--noise", type=int, default=30, \
					 help="Noise training interval")
	parser.add_argument("--patch_size", "--p", type=int, default=64, help="Patch size")
	parser.add_argument("--q", type=int, default=25, \
						help="Q value for jpeg compression (min max)")
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	parser.add_argument("--estimate_parameter_model", type=str, default='./pretrained_models/DPEN_pretrained.pth', \
						help="Pretrained model to estimate distortion parameters")
	argspar = parser.parse_args()

	print("\n### Testing DPEN model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')
	main(**vars(argspar))
	
	

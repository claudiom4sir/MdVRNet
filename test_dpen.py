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

	ps = args['patch_size']

	# load datasets
	valset = DPENDataset(args['valset_dir'], min_sigma=args['sigma'], max_sigma=args['sigma'], min_q=args['q'], max_q=args['q'], patch_size=ps, is_test=True)
	val_dl = DataLoader(valset)

	# create DPEN model and set training params
	model = DPEN()
	model.load_state_dict(torch.load(args['DPEN_model']))
	model = model.cuda()

	# evaluate the model
	sigma = []
	q = []
	model.eval()
	with torch.no_grad():
		for _, data in enumerate(val_dl):
			data, _, _ = data
			data = data.cuda()
			local_est_sigma = []
			local_est_q = []
			_, _, H, W = data.shape
			for h in range((H % ps) // 2, H - ps, ps):
				for w in range((H % ps) // 2, W - ps, ps):
					patch = data[:, :, h:h + ps, w:w + ps]
					estimated_noisestd, estimated_q = model(patch)
					local_est_sigma.append(float(estimated_noisestd[0]))
					local_est_q.append(float(estimated_q[0]))
			sigma.append(np.mean(local_est_sigma))
			q.append(np.mean(local_est_q))
	sigma = np.array(sigma)
	q = np.array(q)
	mae_sigma = np.mean(np.abs(sigma * 255 - args['sigma']))
	mae_q = np.mean(np.abs(q * 100 - args['q']))
	print('MAE for sigma: %f, MAE for q: %f' % (mae_sigma, mae_q))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Test DPEN")
	parser.add_argument("--random_seed", type=int, default=0, help="Random seed to ensure reproducibility")
	parser.add_argument("--sigma", type=int, default=30, \
					 help="Sigma value for AWGN")
	parser.add_argument("--patch_size", "--p", type=int, default=64, help="Patch size")
	parser.add_argument("--q", type=int, default=25, \
						help="Q value for jpeg compression")
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	parser.add_argument("--DPEN_model", type=str, default='./pretrained_models/DPEN_pretrained.pth', \
						help="Pretrained DPEN model to estimate distortion parameters")
	argspar = parser.parse_args()

	print("\n### Testing DPEN model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')
	main(**vars(argspar))
	
	

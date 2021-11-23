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
	# create log dir if it doesn't exist
	if not os.path.isdir(args['log_dir']):
		os.mkdir(args['log_dir'])
	
	# load datasets
	trainset = DPENDataset(args['trainset_dir'], min_sigma=args['noise'][0], max_sigma=args['noise'][1], min_q=args['q'][0], max_q=args['q'][1], patch_size=args['patch_size'])
	train_dl = DataLoader(trainset, args['batch_size'], True)
	valset = DPENDataset(args['valset_dir'], min_sigma=args['noise'][0], max_sigma=args['noise'][1], min_q=args['q'][0], max_q=args['q'][1], patch_size=args['patch_size'])
	val_dl = DataLoader(valset)

	# create DPEN model and set training params
	model = DPEN()
	model = model.cuda()
	current_lr = args['lr']
	criterion = nn.L1Loss(reduction='sum').cuda()
	optimizer = optim.Adam(model.parameters(), lr=current_lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=20)

	# start training DPEN
	best_loss = float("inf")
	for epoch in range(1, args['epochs']):

		# train the model
		model.train()
		train_loss = 0
		for _, data in enumerate(train_dl):
			data, sigma, q = data
			data = data.cuda()
			sigma = sigma.cuda().squeeze(1)
			q = q.cuda().squeeze(1)
			optimizer.zero_grad()
			out_sigma, out_q = model(data)
			loss = criterion(out_sigma, sigma) + criterion(out_q, q)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_loss /= len(trainset)

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

		print("Epoch:%d, train_loss:%f, val_loss:%f, lr:%f, sigma_error:%f, q_error:%f" % (epoch, train_loss, val_loss, current_lr, val_sigma_loss*255., val_q_loss*100))
		scheduler.step(val_loss)

		# save model only if the val loss is decreased from the previous epoch
		if val_loss < best_loss:
			best_loss = val_loss
			torch.save(model.state_dict(), os.path.join(args['log_dir'], 'DPEN.pth'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train DPEN")
	parser.add_argument("--random_seed", type=int, default=0, help="Random seed to ensure reproducibility")
	parser.add_argument("--batch_size", type=int, default=32, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=500, \
					 help="Number of total training epochs")
	parser.add_argument("--lr", type=float, default=1e-4, \
					 help="Initial learning rate")
	parser.add_argument("--noise", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--patch_size", "--p", type=int, default=64, help="Patch size")
	parser.add_argument("--q", type=int, nargs=2, default=[15, 35], \
						help="Q value for jpeg compression (min max)")
	parser.add_argument("--trainset_dir", type=str, default=None, \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path in which to save the model')
	argspar = parser.parse_args()

	print("\n### Training DPEN model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')
	main(**vars(argspar))
	
	

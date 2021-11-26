"""
Trains a FastDVDnet model.
Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>
This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from estimate_params import DPEN
from models import MdVRNet
from dataset import ValDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from mdvrnet import denoise_decompress_seq_mdvrnet
from dataloaders import train_dali_loader
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment, apply_jpeg_artifacts
from train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log_noise_compression, save_model_checkpoint

def main(**args):
	r"""Performs the main training loop
	"""

	# set seed for reproducibility
	seed = args['random_seed']
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	# Load dataset
	print('> Loading datasets ...')
	if args['valset_dir'] is not None:
		dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)

	loader_train = train_dali_loader(batch_size=args['batch_size'],\
									file_root=args['trainset_dir'],\
									sequence_length=args['temp_patch_size'],\
									crop_size=args['patch_size'],\
									epoch_size=args['max_number_patches'],\
									random_shuffle=True,\
									temp_stride=3)
	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)


	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = MdVRNet()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()

	print("Loading DPEN model...")
	estimate_parameter_model = DPEN().cuda()
	estimate_parameter_model.load_state_dict(torch.load(args["DPEN_model"]))
	estimate_parameter_model.eval()
	print("Estimating parameters with model at path %s" % args['DPEN_model'])

	# Define loss
	criterion_mse = nn.MSELoss(reduction='sum').cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	best_psnr_on_val = 0
	is_best = False
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('\nlearning rate %f' % current_lr)

		# train
		epoch_loss = 0.
		for i, data in enumerate(loader_train, 0):

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			img_train, gt_train = normalize_augment(data[0]['data'], ctrl_fr_idx)
			gt_train = gt_train.clone().detach()
			N, C, H, W = img_train.size()

			# std dev of each sequence
			stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['sigma'][0], to=args['sigma'][1])
			# draw noise samples from std dev tensor
			noise = torch.zeros_like(img_train)
			noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

			# define noisy input
			imgn_train = img_train + noise
			imgn_train = torch.clamp(imgn_train, 0., 1.)

			# apply jpeg compression
			min_q = args['q'][0]
			max_q = args['q'][1]
			q = torch.randint(min_q, max_q + 1, (1, N)).squeeze()
			for batch in range(N):
				for frame in range(0, C, 3):
					imgn_train[batch, frame:frame + 3, :, :] = apply_jpeg_artifacts(imgn_train[batch, frame:frame + 3, :, :], q=int(q[batch]))
			imgn_train = torch.clamp(imgn_train, 0., 1.)
			q = q.float() / 100.

			noise_map = torch.zeros((N, 2, H, W))
			# create the noise map if it has to be used
			with torch.no_grad():
				stdn, q = estimate_parameter_model(imgn_train[:, 6:9, :, :])
				stdn = stdn.reshape(N, 1, 1, 1)
			# fill noise map with the std used to blur the corresponding sequence
			for j, current_std in enumerate(stdn[:, 0, 0, 0]):
				noise_map[j, 0].fill_(current_std)
			for j, current_q in enumerate(q):
				noise_map[j, 1].fill_(current_q)

			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			noise_map = noise_map.cuda(non_blocking=True)

			# Evaluate model and optimize it
			out_train = model(imgn_train, noise_map)

			# Compute loss
			loss = criterion_mse(gt_train, out_train)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		print('[Epoch_loss] %f' % (epoch_loss / (i + 1)))

		# Call to model.eval() to correctly set the BN layers before inference
		if args['valset_dir'] is not None:
			model.eval()
			print('Validating the model on the validation set...')
			# Validation and log images
			psnr_val = validate_and_log_noise_compression(
						model_temp=model, \
						dataset_val=dataset_val, \
						valnoisestd=(args['sigma'][0], args['sigma'][1]), \
						valq=(args['q'][0], args['q'][1]), \
						temp_psz=args['temp_patch_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger, \
						trainimg=img_train,
						dpen_model=estimate_parameter_model, dpen_patch=args['patch_size']
						)
			if psnr_val > best_psnr_on_val:
				best_psnr_on_val = psnr_val
				is_best = True
		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch, is_best)
		is_best = False

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train MdVRNet")

	#Training parameters
	parser.add_argument("--random_seed", type=int, default=0, help="Random seed to ensure reproducibility")
	parser.add_argument("--batch_size", type=int, default=32, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=8, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[5, 7], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--DPEN_model", type=str, default='./pretrained_models/DPEN_pretrained.pth', \
						help="Pretrained DPEN model to estimate distortion parameters")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=1,\
						help="Number of training epochs to save state")
	parser.add_argument("--sigma", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=64, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	parser.add_argument("--q", type=int, nargs=2, default=[15, 35], \
						help="Q value for jpeg compression (min max)")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default=None, \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default=None, \
						 help='path of validation set')
	argspar = parser.parse_args()

	print("\n### Training MdVRNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	argspar.sigma[0] /= 255.
	argspar.sigma[1] /= 255.

	main(**vars(argspar))

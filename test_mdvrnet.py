#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import cv2
import torch
import numpy as np
import torchvision as tv
import random
import torch.nn as nn
from estimate_params import DPEN
from models import MdVRNet
from mdvrnet import denoise_decompress_seq_mdvrnet
from utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger, batch_ssim, apply_jpeg_artifacts

NUM_IN_FR_EXT = 5 # temporal size of patch
OUTIMGEXT = '.png' # output images format

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, qval, suffix, save_noisy):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('s{}_q_{}_noisy_{}').format(sigmaval, qval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('s{}_q{}_MdVRNet_{}').format(sigmaval, qval, idx) + fext)
		else:
			out_name = os.path.join(save_dir,\
					('s{}_q{}_MdVRNet_{}_{}').format(sigmaval, qval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def test_fastdvdnet(**args):
	"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_file": path to model
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
			"gray": if True, perform denoising of grayscale images instead of RGB
	"""
	# Start time
	start_time = time.time()

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# Create models
	print('Loading models ...')
	model_temp = MdVRNet(num_input_frames=NUM_IN_FR_EXT)

	# Load saved weights
	state_temp_dict = torch.load(args['model_file'], map_location=device)
	if args['cuda']:
		device_ids = [0]
		model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
	else:
		# CPU mode: remove the DataParallel wrapper
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_temp.load_state_dict(state_temp_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model_temp.eval()

	print("Loading DPEN model...")
	dpen_model = DPEN().cuda()
	dpen_model.load_state_dict(torch.load(args["estimate_parameter_model"]))
	dpen_model.eval()
	dpen_patches = int(args['dpen_patches'])

	with torch.no_grad():
		# process data
		seq, _, _ = open_sequence(args['test_path'],\
									args['gray'],\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
		seq = torch.from_numpy(seq).to(device)

		seqn = seq.clone().detach()

		# Add noise
		print('Adding noise')
		noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
		seqn = seqn + noise
		seqn = torch.clamp(seqn, 0., 1.)

		q = args['q']

		print('Adding compression artifacts')
		for frame in range(0, seq.shape[0]):
			seqn[frame, :, :, :] = apply_jpeg_artifacts(seqn[frame, :, :, :], q=q)
		seqn = torch.clamp(seqn, 0., 1.)

		noisestd = []
		q = []
		for i in range(len(seqn)):
			frame = seqn[i].cpu()
			value_sigma = []
			value_q = []
			_, H, W = frame.shape
			for h in range((H % dpen_patches) // 2, H - dpen_patches, dpen_patches):
				for w in range((H % dpen_patches) // 2, W - dpen_patches, dpen_patches):
					patch = frame[:, h:h + dpen_patches, w:w + dpen_patches]
					estimated_noisestd, estimated_q = dpen_model(patch.unsqueeze(0).cuda())
					value_sigma.append(float(estimated_noisestd[0]))
					value_q.append(float(estimated_q[0]))
			value_sigma = np.mean(value_sigma)
			value_q = np.mean(value_q)		
			noisestd.append(value_sigma)
			q.append(value_q)
		seq_time = time.time()
		denframes = denoise_decompress_seq_mdvrnet(seq=seqn, \
													  noise_std=noisestd, \
													  temp_psz=NUM_IN_FR_EXT, \
													  model_temporal=model_temp, q=q)
	# Compute PSNR and log it
	stop_time = time.time()
	print()
	psnr = batch_psnr(denframes, seq, 1.)
	psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
	ssim = batch_ssim(denframes, seq, 1.)
	ssim_noisy = batch_ssim(seqn.squeeze(), seq, 1.)
	loadtime = (seq_time - start_time)
	runtime = (stop_time - seq_time)
	
	seq_length = seq.size()[0]
	logger.info("Finished restoring {}".format(args['test_path']))
	logger.info("\tRestored {} frames in {:.3f}s, loaded seq in {:.3f}s".\
				 format(seq_length, runtime, loadtime))
	logger.info("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))
	logger.info("\tSSIM noisy {:.4f}dB, SSIM result {:.4f}dB".format(ssim_noisy, ssim))
	print("PSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))
	print("SSIM noisy {:.4f}dB, SSIM result {:.4f}dB".format(ssim_noisy, ssim))

	# Save outputs
	if not args['dont_save_results']:
		# Save sequence
		save_out_seq(seqn, denframes, args['save_path'], \
					   int(args['noise_sigma']*255), int(args['q']), args['suffix'], args['save_noisy'])

	# close logger
	close_logger(logger)

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,\
						default="./model.pth", \
						help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="./data/rgb/Kodak24", \
						help='path to sequence to denoise')
	parser.add_argument("--estimate_parameter_model", type=str, default='./pretrained_models/DPEN_pretrained.pth', \
						help="Pretrained model to estimate distortion parameters")
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=25, \
						help='max number of frames to load per sequence')
	parser.add_argument("--q", type=int, default=15, \
						help="Q value for jpeg compression")
	parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
	parser.add_argument("--dpen_patches", type=float, default=64, help='patch dim for DPEN')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='./results', \
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',\
						help='perform denoising of grayscale images instead of RGB')

	argspar = parser.parse_args()

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing MdVRNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	seed = 0
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

	test_fastdvdnet(**vars(argspar))

# this script requires ffmpeg

import argparse
import subprocess
import os

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Convert .mp4 in .png")
	parser.add_argument("--input_dir", type=str, default=None, help="Dir containing .mp4 files")
	parser.add_argument("--output_dir", type=str, default=None, help="Output dir")
	argspar = parser.parse_args()

	inp_dir = argspar.input_dir
	out_dir = argspar.output_dir
	videos = os.listdir(inp_dir)
	os.mkdir(out_dir)
	os.mkdir(out_dir+'/train/')
	for i, video in enumerate(videos):
  		new_folder = out_dir + '/train/' + str(i)
  		os.mkdir(new_folder)
  		cmd = 'ffmpeg -i ' + inp_dir + '/' + video + ' ' + new_folder + '/%3d.png'
  		subprocess.check_output(cmd, shell=True, text=True)

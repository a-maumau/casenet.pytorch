import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
from tqdm import tqdm
import scipy as sp
from PIL import Image
import numpy as np

from model import Model, ResidualBlock
from loss import loss_function
from data_loader import get_loader
import pair_transforms

def main(args):
	with torch.cuda.device(args.gpu_device_num):
		CASENet = Model(ResidualBlock, [3, 4, 23, 3], 184)
		CASENet.cuda()
		
		# in the dataset, there is a small image, under 224, 224
		# we need preprocess the image to resize.
		# or you can resize it online
		pair_transform = pair_transforms.PairCompose([ 
			pair_transforms.PairRandomCrop(224),
			pair_transforms.PairRandomHorizontalFlip()])

		input_transform = transforms.Compose([ 
			transforms.ToTensor(), 
			transforms.Normalize((0.485, 0.456, 0.406), 
								 (0.229, 0.224, 0.225))])
		"""
		target_transform = transforms.Compose([ 
			transforms.ToTensor()])
		"""
		
		train_loader = get_loader(img_root=args.train_image_dir,
							 mask_root=args.train_mask_dir,
							 json_path=args.train_json_path, 
							 pair_transform=pair_transform,
							 input_transform=input_transform,
							 target_transform=None,
							 batch_size=args.batch_size,
							 shuffle=True,
							 num_workers=args.num_workers)

		val_loader = get_loader(img_root=args.val_image_dir,
							 mask_root=args.val_mask_dir,
							 json_path=args.val_json_path, 
							 pair_transform=pair_transform,
							 input_transform=input_transform,
							 target_transform=None,
							 batch_size=args.batch_size,
							 shuffle=True,
							 num_workers=args.num_workers)
		lr = args.learning_rate
		optimizer = torch.optim.SGD(CASENet.parameters(), lr=lr, momentum=0.9)
		loss_latest = 0

		# Training 
		for epoch in tqdm(range(args.epochs)):
			train_loss_total = 0
			train_prog = tqdm(enumerate(train_loader), total=len(train_loader))
			for i, (images, masks) in train_prog:
				images = Variable(images).cuda()
				masks = Variable(masks).cuda()

				optimizer.zero_grad()
				
				fused_output, side_output = CASENet(images)
				
				# actually, in the edge detection, we need set the weight, witch is none edge pix rate.
				loss_side = loss_function(side_output, masks)
				loss_fuse = loss_function(fused_output, masks)
				loss = loss_side+loss_fuse;
				train_loss_total += loss.data[0]
				loss.backward()
				optimizer.step()

				train_prog.set_description("batch loss : {:.5}".format(loss.data[0]))
				
				torch.save(CASENet.state_dict(), 'CASENet_param_%d.pkl' % (epoch))

				# Decaying Learning Rate
				if (epoch+1) % 30 == 0:
					lr /= 10
					optimizer = torch.optim.SGD(CASENet.parameters(), lr=lr, momentum=0.9)

			print("train loss [epochs {0}/{1}]: {2}".format( epoch, args.epochs,train_loss_total))

			val_prog = tqdm(enumerate(val_loader), total=len(val_loader))
			CASENet.eval()
			val_loss_total=0

			for i, (images, masks) in val_prog:
				images = Variable(images).cuda()
				masks = Variable(masks).cuda()

				optimizer.zero_grad()
				
				fused_output, side_output = CASENet(images)
				
				# actually, in the edge detection, we need set the weight, witch is none edge pix rate.
				loss_side = loss_function(side_output, masks)
				loss_fuse = loss_function(fused_output, masks)
				loss = loss_side+loss_fuse;
				val_loss_total += loss.data[0]

				val_prog.set_description("validation batch loss : {:.5}".format(loss.data[0]))
				if i == 0:
					predic = F.log_softmax(fused_output)
					predic = predic[0]
					_ , ind = predic.sort(1)
					ind = ind.cpu().data.numpy()
					msk = masks.cpu().data.numpy()
					#sp.misc.imsave('output.jpg', ind[0][-1])
					print(ind.shape)
					print(msk)
					ind = Image.fromarray(np.uint8(ind[-1]))
					msk = Image.fromarray(np.uint8(msk[0]))
					ind.save("output_epoch{}.png".format(epoch))
					msk.save("mask_epoch{}.png".format(epoch))

			print("validation loss : {0}".format(val_loss_total))
			CASENet.train()

		print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
		
		# Save the Model
		torch.save(CASENet.state_dict(), 'CASENet_{0}_fin.pkl'.format(args.epochs))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--train_image_dir', type=str, default='./data/train',
						help='directory for train images')
	parser.add_argument('--train_mask_dir', type=str, default='./data/train',
						help='directory for train mask images')
	
	parser.add_argument('--val_image_dir', type=str, default='./data/val',
						help='directory for val images')
	parser.add_argument('--val_mask_dir', type=str, default='./data/val',
						help='directory for validation mask images')
	
	parser.add_argument('--train_json_path', type=str, default='./data/json',
						help='directory of json file for training dataset')
	parser.add_argument('--val_json_path', type=str, default='./data/json',
						help='directory of json file for validation dataset')
	
	parser.add_argument('--crop_size', type=int, default=224,
						help='size for image after processing')

	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--learning_rate', type=float, default=0.01)
	parser.add_argument('--gpu_device_num', type=int, default=0)
	args = parser.parse_args()
	main(args)
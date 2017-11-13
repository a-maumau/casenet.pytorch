import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json
import re
import numpy as np

from PIL import Image

class Dataset(data.Dataset):
	def __init__(self, img_root, mask_root, json_path, pair_transform=None, input_transform=None, target_transform=None):
		self.img_root = img_root
		self.mask_root = mask_root
		
		with open(os.path.join(json_path), "r") as json_file:
			self.images = json.load(json_file)["images"]

		self.input_transform = input_transform
		self.target_transform = target_transform
		self.pair_transform = pair_transform
		self.data_num = len(self.images)
		self.img = []
		self.mask_img = []

		
		for i in range(self.data_num):
			# save as num py array
			_img = np.asarray(Image.open(os.path.join(self.img_root,self.images[i]["file_name"])).convert('RGB'))
			_img.flags.writeable = True
			_img = Image.fromarray(np.uint8(_img))
			self.img.append(_img)
			
			# same file name but it is .png
			_mask_img = np.asarray(Image.open(os.path.join(self.mask_root,re.sub(r'.jpg', "",self.images[i]["file_name"])+".png")))
			_mask_img.flags.writeable = True
			_mask_img = Image.fromarray(np.uint8(_mask_img))
			self.mask_img.append(_mask_img)
			
			#self.mask_img.append(Image.open( os.path.join(self.mask_root,re.sub(r'.jpg', "",self.images[i]["file_name"])+".png")))
	

	def __getitem__(self, index):
		"""Returns one data pair (image and caption)."""

		# Using PIL to open images is very slow? i think... if you don't have enough memory you should use here
		#_img = Image.open(os.path.join(self.img_root,self.images[index]["file_name"])).convert('RGB')
		#_mask_img = Image.open( os.path.join(self.mask_root,re.sub(r'.jpg', "",self.images[index]["file_name"])+".png"))

		if self.pair_transform is not None:
			#_img, _mask_img = self.pair_transform(_img, _mask_img)
			_img, _mask_img = self.pair_transform(self.img[index], self.mask_img[index])
		
		if self.input_transform is not None:
			_img = self.input_transform(_img)
		
		if self.target_transform is not None:
			_mask_img = self.target_transform(_mask_img)
		else:
			_mask_img = torch.from_numpy(np.asarray(_mask_img)).type(torch.LongTensor)

		return _img, _mask_img

	def __len__(self):
		return self.data_num

def collate_fn(data):
	_img, _mask_img = zip(*data)
	_img = torch.stack(_img, 0)
	_mask_img = torch.stack(_mask_img, 0)

	return _img, _mask_img


def get_loader(img_root, mask_root, json_path, pair_transform, input_transform, target_transform, batch_size, shuffle, num_workers):
	dataset = Dataset(img_root=img_root,
					   mask_root=mask_root,  
					   json_path=json_path,
					   pair_transform=pair_transform,
					   input_transform=input_transform,
					   target_transform=target_transform)
	
	data_loader = torch.utils.data.DataLoader(dataset=dataset, 
											  batch_size=batch_size,
											  shuffle=shuffle,
											  num_workers=num_workers,
											  collate_fn=collate_fn)
	return data_loader

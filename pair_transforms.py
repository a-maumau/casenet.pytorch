"""
	wrapping the code of Pytorch
	to perform pair consistent of random values.
	it might be slower than the original code?
	I only wrapped the things I needed.
"""

import torch
import math
import random
import os
from PIL import Image, ImageOps
import numbers

class PairCompose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, input_img, target_img):
		for t in self.transforms:
			input_img, target_img = t(input_img, target_img)
		return input_img, target_img
"""
class PairResize(object):
	def __init__(self, size):
		self.size = size
	def __call__(self, input_img, target_img):
    	return image.resize(size, Image.ANTIALIAS)
"""

class PairRandomHorizontalFlip(object):
	def __call__(self, input_img, target_img):
		if random.random() < 0.5:
			return input_img.transpose(Image.FLIP_LEFT_RIGHT), target_img.transpose(Image.FLIP_LEFT_RIGHT)
		return input_img, target_img

class PairRandomCrop(object):
	def __init__(self, size, padding=0):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding

	def __call__(self, input_img, target_img):
		if self.padding > 0:
			input_img = ImageOps.expand(input_img, border=self.padding, fill=0)
			target_img = ImageOps.expand(target_img, border=self.padding, fill=0)

		# assuming input_img and target_img has same sizw
		w, h = input_img.size
		th, tw = self.size
		if w == tw and h == th:
			return input_img, target_img

		x1 = random.randint(0, w - tw)
		y1 = random.randint(0, h - th)
		return input_img.crop((x1, y1, x1 + tw, y1 + th)), target_img.crop((x1, y1, x1 + tw, y1 + th))

class PairRandomSizedCrop(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		self.size = size
		self.interpolation = interpolation

	def __call__(self, input_img, target_img):
		for attempt in range(10):
			area = img.size[0] * img.size[1]
			target_area = random.uniform(0.08, 1.0) * area
			aspect_ratio = random.uniform(3. / 4, 4. / 3)

			w = int(round(math.sqrt(target_area * aspect_ratio)))
			h = int(round(math.sqrt(target_area / aspect_ratio)))

			if random.random() < 0.5:
				w, h = h, w

			# assuming input_img and target_img has same sizw
			if w <= img.size[0] and h <= img.size[1]:
				x1 = random.randint(0, img.size[0] - w)
				y1 = random.randint(0, img.size[1] - h)

				input_img = img.crop((x1, y1, x1 + w, y1 + h))
				target_img = img.crop((x1, y1, x1 + w, y1 + h))
				assert(target_img.size == (w, h))
				assert(target_img.size == (w, h))

				return input_img.resize((self.size, self.size), self.interpolation), target_img.resize((self.size, self.size), self.interpolation)

		# Fallback
		scale = Scale(self.size, interpolation=self.interpolation)
		crop = CenterCrop(self.size)

		return crop(scale(input_img)), crop(scale(target_img))
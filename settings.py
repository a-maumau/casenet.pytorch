import torchvision.transforms as transforms
from data_loader import get_loader
import pair_transforms

"""
	variavle settings
"""

class_num = 184

# in the dataset, there is a small image, under 224, 224
# we need preprocess the image to resize.
# or you can resize it online
pair_transform = pair_transforms.PairCompose([ 
	pair_transforms.PairRandomCrop(224),
	pair_transforms.PairRandomHorizontalFlip()])

val_pair_transform = pair_transforms.PairCompose([ 
	pair_transforms.PairRandomCrop(224)
	])

input_transform = transforms.Compose([ 
	transforms.ToTensor(), 
	transforms.Normalize((0.485, 0.456, 0.406), 
						 (0.229, 0.224, 0.225))])
"""
target_transform = transforms.Compose([ 
	transforms.ToTensor()])
"""
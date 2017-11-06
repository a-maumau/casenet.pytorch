import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json

from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, img_root, mask_root, json_path, pair_transform=None, input_transform=None, target_transform=None):
        self.img_root = img_root
        self.mask_root = mask_root
        json_file = open(os.path.join(json_path), "r")
        self.images = json.load(json_file)["images"]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.pair_transform = pair_transform
        self.data_num = len(self.images)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        img = Image.open(os.path.join(self.img_root,self.images[index]["file_name"])).convert('RGB')
        mask_img = Image.open( os.path.join(self.mask_root,re.sub(r'.jpg', "",self.images[index]["file_name"])+".png"))

        if self.pair_transform is not None:
            img, mask_img = self.pair_transform(img, mask_img)
        
        if self.input_transform is not None:
            img = self.input_transform(img)
        
        if self.target_transform is not None:
            mask_img = self.target_transform(mask_img)
        else:
            mask_img = torch.from_numpy(np.asarray(mask_img)).type(torch.LongTensor)

        return img, mask_img

    def __len__(self):
        return self.data_num

def collate_fn(data):
    img, mask_img = zip(*data)
    img = torch.stack(img, 0)
    mask_img = torch.stack(mask_img, 0)

    return img, mask_img


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

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
from tqdm import tqdm

from model import Model
from loss import loss_function
from data_loader import get_loader
import pair_transforms

def main(args):
    with torch.cuda.device(1):
        CASENet = Model(ResidualBlock, [3, 4, 23, 3])
        CASENet.cuda()
        
        # in the dataset, there is a small image, under 224, 224
        # I need preprocess the image to resize. 
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
                             json_path=args.json_path, 
                             pair_transform=pair_transform,
                             input_transform=input_transform,
                             target_transform=None,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4)

        lr = args.learning_rate
        optimizer = torch.optim.SGD(resnet.parameters(), lr=lr, momentum=0.9)
        loss_latest = 0

        # Training 
        for epoch in tqdm(range(args.epochs)):
            train_prog = tqdm(enumerate(train_loader), total=5000)
            for i, (images, masks) in train_prog:
                images = Variable(images).cuda()
                masks = Variable(masks).cuda()

                optimizer.zero_grad()
                outputs = CASENet(images)
                outputs = F.log_softmax(outputs)
                loss = loss_function(outputs, masks)
                loss.backward()
                optimizer.step()

                train_prog.set_description("loss : {:.5}".format(loss.data[0]))
                
                torch.save(CASENet.state_dict(), 'CASENet_param_%d.pkl' % (epoch))

                # Decaying Learning Rate
                if (epoch+1) % 20 == 0:
                    lr /= 3
                    optimizer = torch.optim.SGD(CASENet.parameters(), lr=lr, momentum=0.9)

        print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        
        # Save the Model
        torch.save(CASENet.state_dict(), 'CASENet_{0}_fin.pkl'.format(args.epochs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default='./data/train',
                        help='directory for train images')
    parser.add_argument('--train_mask_dir', type=str, default='./data/train',
                        help='directory for train images')
    parser.add_argument('--val_image_dir', type=str, default='./data/val',
                        help='directory for val images')
    parser.add_argument('--json_path', type=str, default='./data/json',
                        help='directory for saving resized images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for image after processing')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
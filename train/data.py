import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset
import split_fcn

def load_dataset(img_path, mask_path, img_size, test_num=2000, load=False):
    transform = transforms.Compose([transforms.ToTensor()])
    # img_path="./Crop_new/" + str(image_size) + "/Image2"
    # mask_path="./Crop_new/" + str(image_size) + "/Mask2"

    if load:
        test_img_lung, test_img_lists, train_img_lung, train_img_lists = \
            split_fcn.split_test_set(img_path, mask_path, test_num)
    else:
        test_img_lung, test_img_lists, train_img_lung, train_img_lists = \
            split_fcn.load_dataset(img_path)
        
    test_set = CustomDataset(img_path=img_path, mask_path=mask_path, \
        img_lists=test_img_lists, mask_lists=test_img_lists, img_size=img_size, transforms=transform)
    train_set = CustomDataset(img_path=img_path, mask_path=mask_path, \
        img_lists=train_img_lists, mask_lists=train_img_lists, img_size=img_size, transforms=transform)
    train_set, val_set = torch.utils.data.random_split(train_set, \
        [len(train_set)-int(len(train_set)/10), int(len(train_set)/10)])

    return train_set, val_set, test_set


def data_loader(dataset, size):
    loader = DataLoader(dataset, batch_size=size, shuffle=True)
    
    return loader
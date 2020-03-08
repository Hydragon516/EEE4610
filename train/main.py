from data import load_dataset, data_loader
from model import UnetGenerator
from utils import save_result
from train import train
import split_fcn
import torch
import glob

root = "./Crop_new/*"
folder = sorted(glob.glob(root))

for path in folder:
    image_size = int(path.split("/")[2])
    image_path = "./Crop_new/" + str(image_size) + "/Image2"
    mask_path = "./Crop_new/" + str(image_size) + "/Mask2"
    
    train_set, val_set, test_set = load_dataset(img_path=image_path, mask_path=mask_path, img_size=image_size, test_num=2000, load=False)
    train_loader = data_loader(train_set, 16)
    val_loader = data_loader(val_set, 16)
    
    net = UnetGenerator(in_dim=1,out_dim=1,num_filter=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    test = train(net=net, image_size=image_size, train_loader=train_loader, val_loader=val_loader)
    test.run(learning_rate=0.001, patience=10, mini_batch=50, epochs=50, txt_logger=False, model_save=False, early_stopping=True)
    
    test_loader = data_loader(test_set, 4)
    
    save_result(image_size=image_size, data_loader=test_loader, net=net)
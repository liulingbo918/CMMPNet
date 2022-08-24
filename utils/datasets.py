import os
from sklearn.model_selection import train_test_split
from .data_loader import ImageGPSDataset, ImageLidarDataset



def prepare_Beijing_dataset(args):
    print("")
    print("Dataset: ", args.dataset)
    print("down resolution: ", args.down_scale)
        
    print("")
    print("sat_dir: ", args.sat_dir)
    print("gps_dir: ", args.gps_dir)    
    print("mask_dir: ", args.mask_dir)
    
    print("")
    print("test_sat_dir: ", args.test_sat_dir)
    print("test_gps_dir: ", args.test_gps_dir)    
    print("test_mask_dir: ", args.test_mask_dir)
    print("")
    
    image_list = [x[:-9] for x in os.listdir(args.mask_dir)      if x.find('mask.png') != -1]
    test_list  = [x[:-9] for x in os.listdir(args.test_mask_dir) if x.find('mask.png') != -1]
    train_list, val_list = train_test_split(image_list, test_size=args.val_size, random_state=args.random_seed)
    
    train_dataset = ImageGPSDataset(train_list, args.sat_dir,      args.mask_dir,      args.gps_dir,      randomize=True,  down_scale=args.down_scale)
    val_dataset   = ImageGPSDataset(val_list,   args.sat_dir,      args.mask_dir,      args.gps_dir,      randomize=False, down_scale=args.down_scale)
    test_dataset  = ImageGPSDataset(test_list,  args.test_sat_dir, args.test_mask_dir, args.test_gps_dir, randomize=False, down_scale=args.down_scale)

    return train_dataset, val_dataset, test_dataset
    
    
def prepare_TLCGIS_dataset(args):
    print("")
    print("Dataset: ", args.dataset)
    mask_transform = True if args.dataset == 'TLCGIS' else False
    adjust_resolution =512 if args.dataset == 'TLCGIS' else -1
    
    print("")
    print("sat_dir: ", args.sat_dir)
    print("gps_dir: ", args.lidar_dir)    
    print("mask_dir: ", args.mask_dir)
    print("partition_txt: ", args.split_train_val_test)
    print("mask_transform: ", mask_transform)
    print("adjust_resolution: ", adjust_resolution)
    print("")
        
    train_list = val_list = test_list = []
    with open(os.path.join(args.split_train_val_test,'train.txt'),'r') as f:
        train_list = [x[:-1] for x in f]
    with open(os.path.join(args.split_train_val_test,'valid.txt'),'r') as f:
        val_list = [x[:-1] for x in f]
    with open(os.path.join(args.split_train_val_test,'test.txt'),'r') as f:
        test_list = [x[:-1] for x in f]

    train_dataset = ImageLidarDataset(train_list, args.sat_dir, args.mask_dir, args.lidar_dir, randomize=False,  mask_transform=mask_transform, adjust_resolution=adjust_resolution)
    val_dataset   = ImageLidarDataset(val_list,   args.sat_dir, args.mask_dir, args.lidar_dir, randomize=False, mask_transform=mask_transform, adjust_resolution=adjust_resolution)
    test_dataset  = ImageLidarDataset(test_list,  args.sat_dir, args.mask_dir, args.lidar_dir, randomize=False, mask_transform=mask_transform, adjust_resolution=adjust_resolution)

    return train_dataset, val_dataset, test_dataset



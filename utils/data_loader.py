import os
import cv2
import torch
import torch.utils.data as data
from .image_augmentation import *
#from .gps_render import GPSDataRender, GPSImageRender
#from skimage.segmentation import slic

class ImageLidarDataset(data.Dataset):
    def __init__(self, image_list, sat_root, mask_root, lidar_root, sat_suffix="png", mask_suffix="png", lidar_suffix="png", randomize=False, mask_transform=False, adjust_resolution=-1):
        self.image_list = image_list
        
        self.sat_root   = sat_root
        self.mask_root  = mask_root
        self.lidar_root = lidar_root
        
        self.sat_suffix   = sat_suffix
        self.mask_suffix  = mask_suffix
        self.lidar_suffix = lidar_suffix
        
        self.randomize = randomize
        self.mask_transform = mask_transform
        self.adjust_resolution = adjust_resolution
        
    def _read_data(self, image_id):
        img   = cv2.imread(os.path.join(self.sat_root,   "{0}.{1}").format(image_id, self.sat_suffix))   
        mask  = cv2.imread(os.path.join(self.mask_root,  "{0}.{1}").format(image_id, self.mask_suffix),  cv2.IMREAD_GRAYSCALE)
        lidar = cv2.imread(os.path.join(self.lidar_root, "{0}.{1}").format(image_id, self.lidar_suffix), cv2.IMREAD_GRAYSCALE)
        
        assert (img is not None),   os.path.join(self.sat_root,   "{0}.{1}").format(image_id, self.sat_suffix)
        assert (mask is not None),  os.path.join(self.mask_root,  "{0}.{1}").format(image_id, self.mask_suffix)
        assert (lidar is not None), os.path.join(self.lidar_root, "{0}.{1}").format(image_id, self.lidar_suffix)
        
        ## In TLCGIS, the foreground value is 0 and the background value is 1.
        ## The background value is transformed to 0 and the foreground value is transformed to 1 (255)
        if self.mask_transform:
           mask = (1 - mask) * 255

        return img, mask, lidar

    
    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, sat, mask, lidar):
        if lidar.ndim == 2:
            lidar = np.expand_dims(lidar, axis=2)
            
        if self.randomize:
            sat = randomHueSaturationValue(sat)   
            img = self._concat_images(sat, lidar)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        else:
            img = self._concat_images(sat, lidar)

        if mask.ndim == 2:
           mask = np.expand_dims(mask, axis=2)
        
        # The image's resolution of TLCGIS is 500*500. We change the resolution of input images to 512*512 due to the requirements of network structure.
        # But the resolution of masks is maintained. For a fair comparison, the final predicted maps would be resized to the resolution of masks during testing.
        if self.adjust_resolution > 0:
           img = cv2.resize(img, (self.adjust_resolution, self.adjust_resolution))
            
        try:
            img  = np.array(img,  np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)
        
        mask[mask >= 0.5] = 1
        mask[mask <  0.5] = 0
        return img, mask    

    def __getitem__(self, index):
        image_id = self.image_list[index]
        img, mask, lidar = self._read_data(image_id)
        img, mask = self._data_augmentation(img, mask, lidar)
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.image_list)



class ImageGPSDataset(data.Dataset):
    def __init__(self, image_list, sat_root, mask_root, gps_root, sat_suffix="png", mask_suffix="png", pgs_suffix="jpg", randomize=True, down_scale = True, down_resolution=512):
        self.image_list = image_list
        
        self.sat_root = sat_root
        self.mask_root = mask_root
        self.gps_root = gps_root
        
        self.sat_suffix  = sat_suffix
        self.mask_suffix = mask_suffix
        self.gps_suffix  = pgs_suffix
        
        self.randomize = randomize
        self.down_scale = down_scale
        self.down_resolution = down_resolution

    def _read_data(self, image_id):
        img_path  = os.path.join(self.sat_root,  "{0}_sat.{1}").format(image_id,  self.sat_suffix)
        mask_path = os.path.join(self.mask_root, "{0}_mask.{1}").format(image_id, self.mask_suffix)
        gps_path  = os.path.join(self.gps_root,  "{0}_gps.{1}").format(image_id,  self.gps_suffix)
        
        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gps  = cv2.imread(gps_path,  cv2.IMREAD_GRAYSCALE)
        
        assert (img is not None),  img_path
        assert (mask is not None), mask_path
        assert (gps is not None),  gps_path
        
        return img, mask, gps


    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, sat, mask, gps):
        if gps.ndim == 2:
            gps = np.expand_dims(gps, axis=2)
            
        if self.randomize:
            sat = randomHueSaturationValue(sat)
            img = self._concat_images(sat, gps)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomRotate180(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomRotate270(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
            img, mask = randomcrop(img,mask)
        else:
            img = self._concat_images(sat, gps)
    
        # The image's resolution of BJRoad is too high. To reduce memory consumption, we reduce the resolution of input images to 512*512
        # But the resolution of masks is maintained. For a fair comparison, the final predicted maps would be resized to the resolution of masks during testing.
        if self.down_scale:
           img = cv2.resize(img, (self.down_resolution, self.down_resolution))
        
        if mask.ndim == 2:
           mask = np.expand_dims(mask, axis=2)

        try:
            img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)

        mask[mask >= 0.5] = 1
        mask[mask <  0.5] = 0
        return img, mask


    def __getitem__(self, index):
        image_id = self.image_list[index]
        img, mask, gps= self._read_data(image_id) 
        img, mask = self._data_augmentation(img, mask, gps)
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.image_list)


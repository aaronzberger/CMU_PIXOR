'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import numpy as np
import torch
from config import data_dir, base_dir
from utils import load_config
from utils import get_points_in_a_rotated_box, trasform_label2metric
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from copy import deepcopy


class KITTI(Dataset):
    target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
    target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])

    def __init__(self, frame_range=10000, split='train'):
        '''
        KITTI Dataset for PyTorch

        Parameters:
            frame_range (int): maximum KITTI index to use
            split (str): split to use (train, test, etc.)
        '''
        self.frame_range = frame_range

        self.kitti_indices = self.load_split(split)
        self.velo = self.load_velo(split)

        self.geometry = load_config()[0]['geometry']

    def __len__(self):
        return len(self.kitti_indices)

    def __getitem__(self, item):
        # Load and pre-process the point cloud
        scan = np.fromfile(self.velo[item], dtype=np.float32).reshape(-1, 4)
        scan = self.lidar_preprocess(scan)

        # # Save an image of the point cloud
        # image = cv.cvtColor(np.amax(deepcopy(scan), axis=2), cv.COLOR_GRAY2BGR) * 255
        # cv.imwrite(os.path.join(base_dir, 'viz', 'pcl{}.jpg'.format(self.kitti_indices[item])), image)

        scan = torch.from_numpy(scan).permute(2, 0, 1)

        # Create the label_map from the KITTI label
        label_map = self.get_label(item)
        self.reg_target_transform(label_map)
        label_map = torch.from_numpy(label_map)

        # Save an image of the ground truth label map
        # image_truth = np.zeros((self.geometry['label_shape'][0], self.geometry['label_shape'][1], 3), dtype=np.uint8)
        # for p in torch.nonzero(deepcopy(label_map)):
        #     image_truth[np.int64(p[0]), np.int64(p[1])] = (255, 255, 255)
        # cv.imwrite(os.path.join(base_dir, 'viz', 'labels{}.jpg'.format(self.kitti_indices[item])), image_truth)

        label_map = label_map.permute(2, 0, 1)

        return scan, label_map, item

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev


    def load_split(self, split):
        '''
        Load the split file

        Parameters:
            split (string): split file name, before .txt

        Returns:
            list: the indices of the images in the KITTI dataset for this split
        '''
        path = os.path.join(data_dir, '{}.txt'.format(split))

        if not os.path.exists(path):
            raise ValueError('split argument must have a corresponding txt file in {}'.format(data_dir))

        with open(path, 'r') as f:
            lines = f.readlines()
            kitti_indices = []
            for line in lines[:-1]:
                if int(line[:-1]) < self.frame_range:
                    kitti_indices.append(line[:-1])

            # Last line does not have a \n symbol
            last = lines[-1][:6]
            if int(last) < self.frame_range:
                kitti_indices.append(last)

            print('Split file {}: '.format(path), end='')

            return kitti_indices

    def fill_boxes(self, label_map, bev_corners, reg_target):
        '''
        Fill the points inside GT boxes in the label_map

        Parameters:
            label_map (np.ndarray): the ground truth label_map (empty)
            bev_corners (np.ndarray): (4, 2) corners of a BEV bounding box
            reg_target (list): (6): regression target

        Returns:
            np.ndarray: label_map parameter, with filled in boxes

        '''
        label_corners = (bev_corners / 4 ) / 0.1
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2

        # Find all BEV pixels that are in any of the boxes
        points = get_points_in_a_rotated_box(label_corners, self.geometry['label_shape'])

        for p in points:
            label_x = p[0]
            label_y = p[1]
            metric_x, metric_y = trasform_label2metric(np.array(p))
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            label_map[label_y, label_x, 0] = 1.0
            label_map[label_y, label_x, 1:7] = actual_reg_target

        return label_map

    def get_label(self, index):
        '''
        Get the GT label_map

        Parameters:
            index (int): the index of the label to retrieve with respect to
                the kitti_indices list
            
        Returns:
            np.ndarray: label_map, an image with the GT boxes filled in where KITTI labels are
        '''
        index = self.kitti_indices[index]
        filename = (6 - len(index)) * '0' + index + '.txt'
        label_path = os.path.join(data_dir, 'training', 'label_2', filename)

        object_list = {'Car': 0} # Add more objects here if desired
        label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)

        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])

                    if name == 'Car':
                        # Calculate the BEV bounding box corners
                        corners, reg_target = kitti_to_bev_box(bbox)

                        # Fill the GT boxes in the label_map with white pixels
                        label_map = self.fill_boxes(label_map, corners, reg_target)

        return label_map

    def load_velo(self, split):
        '''
        Load the velodyne scan data for this split from the KITTI binary files

        Parameters:
            split (string): split file name, before .txt

        Returns:
            list: the paths to the velodyne files with respect to the kitti_indices class variable
        '''
        velo_files = []
        for i in self.kitti_indices:
            path = os.path.join(data_dir, '{}ing'.format(split), 'velodyne', '{}.bin'.format(i))
            if os.path.exists(path):
                velo_files.append(path)
            else:
                raise ValueError('Failed to find the velodyne scan {}, but it should exist'.format(path))

        print('Successfully found {} out of {} velodyne scans'.format(len(velo_files), len(self.kitti_indices)))

        return velo_files

    def lidar_preprocess(self, scan):
        '''
        Transform the velodyne scan into a BEV image with a Z feature channel

        Parameters:
            scan (np.ndarray): point cloud - [N x [x, y, z, r]] format
        '''
        # Filter the lidar by bounds
        geom = self.geometry
        valid = (geom['W1'] < scan[:, 0]) * (scan[:, 0] < geom['W2']) * \
                (geom['L1'] < scan[:, 1]) * (scan[:, 1] < geom['L2']) * \
                (geom['H1'] < scan[:, 2]) * (scan[:, 2] < geom['H2'])
        indices = np.where(valid)[0]
        velo = scan[indices, :]

        # Make the BEV image
        velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)

        # Intensity map keeps track of how many points there are at an xy position
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))

        for i in range(velo.shape[0]):
            x = int((velo[i, 1] - self.geometry['L1']) / 0.1)
            y = int((velo[i, 0] - self.geometry['W1']) / 0.1)
            z = int((velo[i, 2] - self.geometry['H1']) / 0.1)
            velo_processed[x, y, z] = 1
            velo_processed[x, y, -1] += velo[i, 3]
            intensity_map_count[x, y] += 1

        # Make the last Z space of each XY line the average reflectance:
        #    (sum of reflectances) / (number of points)
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count,
                                             where=intensity_map_count != 0)

        return velo_processed

def kitti_to_bev_box(bbox):
    w, h, l, y, z, x, yaw = bbox[8:15]
    y = -y
    # manually take a negative s. t. it's a right-hand system, with
    # x facing in the front windshield of the car
    # z facing up
    # y facing to the left of driver

    yaw = -(yaw + np.pi / 2)
    
    #x, y, w, l, yaw = self.interpret_kitti_label(bbox)
    
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    # rear left
    bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
    bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

    # rear right
    bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
    bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

    # front right
    bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
    bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

    # front left
    bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
    bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

    reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

    return bev_corners, reg_target


def get_data_loader(frame_range=10000):
    config, _, batch_size, _ = load_config()
    train_dataset = KITTI(frame_range, split='train')
    train_data_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=3)

    test_dataset = KITTI(frame_range, split='test')
    test_data_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=config['val_batch_size'], num_workers=8)

    return train_data_loader, test_data_loader, \
        len(train_data_loader), len(test_data_loader)

import os
from glob import glob
import os.path as osp
import torch
from PIL import Image
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
'''


class SYSUDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

class RegdbDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, trial=1):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = [i for i in range(206)]
            selected_ids = train_ids
        else:
            test_ids = [i for i in range(206)]
            selected_ids = test_ids

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)
        
        train_color_txt   = osp.join(root, 'idx/train_visible_{}'.format(trial)+ '.txt')
        train_thermal_txt = osp.join(root, 'idx/train_thermal_{}'.format(trial)+ '.txt')
        gallery_txt   = osp.join(root, 'idx/test_visible_{}'.format(trial)+ '.txt')
        query_txt = osp.join(root, 'idx/test_thermal_{}'.format(trial)+ '.txt')
        
        def load_data(root, txt_path, visible=True):
            data_file_list = open(txt_path, 'rt').read().splitlines()
            # Get full list of image and labels
            paths = []
            pids = []
            cams = []
            for s in data_file_list:
                image_path = osp.join(root, s.split(' ')[0]) 
                person_id = int(s.split(' ')[1])
                cam_id = 0 if visible else 1
                
                paths.append(image_path)
                pids.append(person_id)
                cams.append(cam_id)
            return paths, pids, cams
        
        if mode == 'train':
            paths_rgb, pids_rgb, cams_rgb = load_data(root, train_color_txt, visible=True)
            paths_ir, pids_ir, cams_ir = load_data(root, train_thermal_txt, visible=False)
            self.img_paths = paths_rgb + paths_ir
            self.ids = pids_rgb + pids_ir
            self.cam_ids = cams_rgb + cams_ir
        elif mode == 'query':
            self.img_paths, self.ids, self.cam_ids = load_data(root, query_txt, visible=False)
        else:
            self.img_paths, self.ids, self.cam_ids = load_data(root, gallery_txt, visible=True)

        self.num_ids = 206
        self.transform = transform


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, path, item

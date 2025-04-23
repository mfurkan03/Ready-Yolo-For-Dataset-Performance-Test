import torch
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset
import os
from PIL import Image
import csv
from torch.utils.data import DataLoader

class CustomYoloDataset(Dataset):
    def __init__(self,set_type, image_dir, label_dir, normalize=False, augment=False):
        assert set_type in {'train', 'test'}

        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.dataset = list(zip(
            [os.path.join(image_dir, f) for f in self.image_files if f.endswith('.jpg')],
            [os.path.join(label_dir, f) for f in self.label_files if f.endswith('.csv')]
        ))
        self.dataset = self.dataset[:int(len(self.dataset) * 0.9)] if set_type == 'train' else self.dataset[int(len(self.dataset) * 0.9):]
        self.normalize = normalize
        self.augment = augment
        self.classes = utils.load_class_dict()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE),
        ])
        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f'Generating class dict')):
                data, label = data_pair
                for j, bbox_pair in enumerate(utils.get_bounding_boxes(self.parse_yolo_label_csv(label))):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            utils.save_class_dict(self.classes)
       

    def parse_yolo_label_csv(self,csv_path, class_to_id=None):
        boxes = []
        object = {"bndbox":{"xmin":0,"xmax":0,"ymin":0,"ymax":0},"name":"str"}
        label = {"annotation": {"object": [], "size": {"width":0,"height":0}}}  # Dummy label structure for compatibility
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:

                obj = {"bndbox":{"xmin":0,"xmax":0,"ymin":0,"ymax":0},"name":"str"}
                obj['name'] = row['class']

                width_img = int(row['width'])
                height_img = int(row['height'])

                # BBox koordinatları
                xmin = int(row['xmin'])
                xmax = int(row['xmax'])
                ymin = int(row['ymin'])
                ymax = int(row['ymax'])         

                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax

                boxes.append(obj)

            label['annotation']['object'] = boxes
            label['annotation']['size']['width'] = width_img
            label['annotation']['size']['height'] = height_img
        
        return label
    
    def __getitem__(self, i):
        data, label = self.dataset[i]
        data = Image.open(data)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        data = self.transform(data)

        label = self.parse_yolo_label_csv(label)
        original_data = data.clone()
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        grid_size_x = data.size(dim=2) / config.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 5 * config.B + config.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],     # Y coord relative to grid square
                            (x_max - x_min) / config.IMAGE_SIZE[0],                 # Width
                            (y_max - y_min) / config.IMAGE_SIZE[1],                 # Height
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + config.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    # Display data
    obj_classes = utils.load_class_array()
    dir = r"C:\Users\alkan\.cache\kagglehub\datasets\a2015003713\militaryaircraftdetectiondataset\versions\87\dataset"
    train_set = DataLoader(CustomYoloDataset(dir, dir, normalize=False, augment=False))
    #train_set = DataLoader(YoloPascalVocDataset('train', normalize=False, augment=False))
    item = next(iter(train_set))
    data, label, original_data = item
    data,label, original_data = data[0], label[0], original_data[0]
    negative_labels = 0
    smallest = 0
    largest = 0
    negative_labels += torch.sum(label < 0).item()
    smallest = min(smallest, torch.min(data).item())
    largest = max(largest, torch.max(data).item())
    utils.plot_boxes(data, label, obj_classes, max_overlap=float('inf'))
    # print('num_negatives', negative_labels)
    # print('dist', smallest, largest)
    # print('dist', smallest, largest)
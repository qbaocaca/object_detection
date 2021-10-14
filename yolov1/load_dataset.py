import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PascalVOC (Dataset):

    def __init__(self, csv_file, img_dir, label_dir, split_size=7,
                 num_boxes=2, num_classes=20, transform=None):

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes


    def __len__ (self):
        return len(self.annotations)

    def __getitem__ (self, index):

        # process the label
        # locate the index row, take the 1st column where the txt file is

        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():

                class_label, x, y, width, height = []

                # the class label is int should be kept as int
                # the rest are float

                for item in label.replace("\n", "").split():
                    if float(item) != int(float(item)):
                        [class_label, x, y, width, height].append(float(item))
                    else:
                        [class_label, x, y, width, height].append(int(item))

                boxes.append([class_label, x, y, width, height])

        # process the image

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        matrix_size = (self.S, self.S, self.C + 5 * self.B)

        # even 30 are specified, only 25 nodes are used
        # 30 is for prediction with two bboxes
        # label only has one

        label_matrix = torch.zeros(matrix_size)

        # convert the bbox relative to the entire image
        # to relative to a particular cell

        # want to know which cell bbox belongs to

        for box in boxes:

            # convert to tensor to do transform
            # convert back to list

            class_label, x, y, width, height = box.toList()
            class_label = int(class_label)

            # to know which cell row and column bboxes belong
            # multiply with the number of cell

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # to get the width and height of bbox compared to the cell
            # multiply with the number of cell

            width_cell, height_cell = self.S * width, self.S * height

            # if there is no obj in the cell

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([
                    x_cell, y_cell, width_cell, height_cell
                ])

                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix



































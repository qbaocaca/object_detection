import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm # how to get a progress bar in pytorch
                      # so that it looks nice!

from torch.utils.data import DataLoader

from pytorch.yolov1.load_dataset import PascalVOC
from pytorch.yolov1.yolov1_model import yolov1
from pytorch.yolov1.yolo_loss import yolo_loss
from pytorch.yolov1.non_max_suppression import _nms
from pytorch.yolov1.mean_average_precision import mAP
from pytorch.object_detection.metrics.iou import intersection_over_union
from pytorch.object_detection.main.utils import load_checkpoint, get_bboxes

learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64 # 64 is used in the paper, preferably 16

# not gonna train the entire model
# just overfit 1 to 100 examples
weight_decay = 0
num_epochs = 100
num_workers = 2
pin_memory = True
load_model = False
load_model_file = "overfit.pth.tar"

img_dir = "put_img_dir_here"
label_dir = "put_label_dir_here"

# create an own class for Compose for doing data augmentation easily

class Compose(object):
    def __init__ (self, transforms):
        self.transforms = transforms

    def __call__ (self, img, bboxes):
        for transform in self.transforms:
            img, bboxes = transform(img), bboxes

        return img, bboxes

    # transform only does on the image

# can also does horizontalflip, normalization

transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

def train_fn (train_loader, model, optimizer, loss_fn):

    # setting a progress bar
    loop = tqdm(train_loader, leave=True)

    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

model = yolov1(in_channels=3, split_size=7, num_boxes=2, num_classes=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)

criterion = yolo_loss(split_size=7, num_boxes=2, num_classes=20)

if load_model:
    load_checkpoint(torch.load(load_model_file), model, optimizer)

train_dataset = PascalVOC(
        csv_file= "put_csv_file_here",
        img_dir=img_dir,
        label_dir=label_dir,
        transform=transform
    )

test_dataset = PascalVOC(
        csv_file="put_csv_file_here",
        img_dir=img_dir,
        label_dir=label_dir,
        transform=transform
    )

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=pin_memory,
                          shuffle=True, drop_last=False)

# if there is a batch doesn't have enough examples, drop it

test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=pin_memory,
                          shuffle=False, drop_last=False)

for epoch in range(num_epochs):

    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4
    )

    mean_average_precision = mAP(pred_boxes=pred_boxes,
                                 true_boxes=target_boxes,
                                 iou_threshold=0.5,
                                 box_format="midpoint",
                                 num_classes=20)

    print(f"Train mAp: {mean_average_precision}")

    train_fn(train_loader, model, optimizer, criterion)























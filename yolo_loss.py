import torch
from pytorch.object_detection.metrics.iou import intersection_over_union
import torch.nn as nn

class yolo_loss (nn.Module):
    def __init__ (self, split_size=7, num_boxes=2, num_classes=20):
        super(yolo, self).__init__()

        # Number of bboxes outputing for each cell is 2.

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.mse = nn.MSELoss(reduction="sum")

        # default reduction is "mean"
        # The mean operation operates over all the elements, and divides by n.
        # The division by n can be avoided if one sets reduction = 'sum'.

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):

        # reshape the tensor if its not in form S*S*30
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 0 to 19 is class prob
        # 20 is for class score, 0 or 1, identity function, whether there is an object
        # in that bbox
        # 21 to 25 is for bbbox value of box_1: 21, 22, 23, 24
        # 26 to 30 is for bbox value of box_2: 26, 27, 28, 29

        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        _, best_iou = torch.max(ious, dim=0)
        exists_box = targets[..., 20].unsqueeze(3)

        # could do exists_box = targets[..., 20:21].unsqueeze(3)
        # means index 20, but want to keep number of dimension ???
        # before dimension of [..., 20:21] is (1, )
        # after .unsqueeze(3), becomes (1,  , 1) ???
        # adding 1 into index 3 of dimension ???

        # best_box is 0 or 1 depending on which box is the best ???

        # calculate loss for box_format = "midpoint"
        # (x, y, w, h) ???
        # identity function step

        box_predictions = exists_box * (

            best_iou * predictions[..., 26:30]
            + (1 - best_iou) * predictions[..., 21:25] # this best_iou is 0 if the 2nd bbox
                                                       # is the best ???
        )

        box_targets = exists_box * targets[..., 21:25]

        # calculate loss for height and width of bboxes
        # taking the absolute in case there is negative
        # plus 1e-6 in case there is 0 because derivative of square root as go to 0 is
        # negative infinity ???
        # make sure the sign of the gradient is correct ???

        sign = torch.sign(box_predictions[..., 2:4])

        box_predictions[..., 2:4] = sign * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # explanation of torch.flatten is below
        # mse loss takes in input and target

        # shape of input: (N, *) where * means, any number of additional dimensions
        # shape of target: (N, *), same shape as the input

        # shape of box loss (N, S, S, 4)
        # after flatten, becomes (N*S*S, 4)
        # flatten from the beginning to the second from last
        # leaving the last dimension

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # calculate the object loss

        pred_box = exists_box * (
            best_iou * predictions[..., 25:26] +
            (1 - best_iou) * predictions[..., 20:21]
        )

        pred_target = exists_box * targets[..., 20:21]

        # don't need to specify start_dim and end_dim here
        # because flatten will automatically flatten everything from index 0
        # (N, S, S, 1) --> (N*S*S*1)

        object_loss = self.mse(
            torch.flatten(pred_box),
            torch.flatten(pred_target)
        )

        # calculate no object loss

        no_pred_box_1 = (1 - exists_box) * predictions[..., 20:21]
        no_target_box_1 = (1 - exists_box) * targets[..., 20:21]
        no_pred_box_2 = (1 - exists_box) * predictions[..., 25:26]
        no_target_box_2 = (1 - exists_box) * targets[..., 25:26]

        # shape (N, S, S, 1) --> (N, S*S*1)

        no_object_loss_box_1 = self.mse(
            torch.flatten(no_pred_box_1, start_dim=1),
            torch.flatten(no_target_box_1, start_dim=1)
        )

        no_object_loss_box_2 = self.mse(
            torch.flatten(no_pred_box_2, start_dim=1),
            torch.flatten(no_target_box_2, start_dim=1)
        )

        no_object_loss = no_object_loss_box_1 + no_object_loss_box_2

        # calculate class loss

        class_pred_box = exists_box * predictions[..., :20]
        class_target_box = exists_box * targets[..., :20]

        # (N, S, S, 20) --> (N*S*S, 20)

        class_loss = self.mse(
            torch.flatten(class_pred_box, end_dim=-2),
            torch.flatten(class_target_box, end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss


# t = torch.tensor([ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ])
# print(t.shape)

# Flattens input by reshaping it into a one-dimensional tensor.
# If start_dim or end_dim are passed, only dimensions starting with start_dim
# and ending with end_dim are flattened.
# The order of elements in input is unchanged.

# start_dim is the first dim to flatten
# flatten to the end of tensor if no end_dim is specified
# start_dim default is 0
# end_dim default is -1

# The negative index is used in python to index
# starting from the last element of the list
# tuple, or any other container class which supports indexing
# -1 refers to the last index, -2 refers to the second last index, and so on.

# t1 = torch.flatten(t, start_dim=1)
# print(t1.shape)














import torch
from collections import Counter
from pytorch.object_detection.metrics.iou import intersection_over_union

def mAP (pred_boxes, true_boxes, iou_threshold=0.5,
         box_format="corners", num_classes=20):

    # pred_boxes: list of all predicted bboxes [[],[],[]]
    # each has 7 items
    # train_idx is the image where the bboxes come from

    average_precisions = []
    epsilon = 1e-6

    for _class in range(num_classes):
        detections = []
        ground_truths = []

        for bboxes in pred_boxes:

            if bboxes[1] == _class:
                detections.append(bboxes)

        for bboxes in true_boxes:

            if bboxes[1] == _class:
                ground_truths.append(bboxes)

        train_idx_list = []

        for box in ground_truths:
            train_idx_list.append(box[0])

        amount_bboxes = Counter(train_idx_list)

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections)) # tensor of zero elements
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # taken out a single bbox for a particular class in a particular image

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):

            # taken out all ground truth bboxes for a particular image

            ground_truth_img = []

            for box in ground_truths:
                if box[0] == detection[0]:
                    ground_truth_img.append(box)

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, box in enumerate(ground_truth_img):

                # compare that bbox in detection with all target_boxes

                pred_box = torch.tensor(detection[3:])
                target_box = torch.tensor(box[3:])

                # check iou of that bbox in detection with all target_boxes

                iou = intersection_over_union(pred_box, target_box,
                                              box_format=box_format)

                # keep track of the best iou

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # if this iou is larger than the threshold, then that bbox is a TP
            # TP is a bbox that share iou > threshold with target bbox
            # TP means bbox detects some parts of the object
            # FP is a bbox that share iou < threshold with target bbox
            # FP means bbox detects some random parts of the image

            if best_iou > iou_threshold:

                # first, check if this bbox has been covered before or not

                # amount_bboxes = {0:3, 1:5, 2:4, 3:5, ...}
                # amount_bboxes = Counter({0: tensor([0., 0., 0.]),
                # 1: tensor([0., 0., 0., 0., 0.]),.......})

                # 0, 1, 2, ... are train_idx indicating the image the bbox belongs
                # amount_bboxes[key] where key is the train_idx
                # amount_bboxes[key] is an image
                # amount_bboxes[0] is tensor([0., 0., 0.]) contains 3 bboxes
                # amount_bboxes[0][0] is tensor(0.) is a bbox

                key = detection[0]

                if amount_bboxes[key][best_gt_idx] == 0:

                    # this bbox has not been covered
                    # because it tensor equals 0. 0 is the initialization

                    # TP has form of a tensor([0., 0., 0.,....])

                    TP[detection_idx] = 1
                    amount_bboxes[key][best_gt_idx] = 1

                    # now that bbox is covered by setting it to 1

                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # TP now will have the form of tensor([0., 1, 0., 1, 1, 0., 0.,....])
        # 1 is correct prediction bbox, 0 is incorrect prediction bbox
        # TP_cumsum = tensor([0., 1, 0., 2, 3, 3, 3, .....])

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)

        # adding a point (0,1) for numrical integration ???
        # precision is y-axis, recall is x-axis

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



t1_preds = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
t1_targets = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
#     [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
# [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
#     [2, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
#     [3, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
#     [3, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

# result = mAP(
#             t1_preds,
#             t1_targets,
#             iou_threshold=0.5,
#             box_format="midpoint",
#             num_classes=1,
#         )

# print(result)
# print("result[0] is ", result[0])
# print("result[0][0] is ", result[0][0])



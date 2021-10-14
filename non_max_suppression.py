import torch
from pytorch.object_detection.metrics.iou import intersection_over_union
from pytorch.object_detection.metrics.nms import nms

a = [[1, 0.1, 0.5, 0.6, 0.8, 0.9],
     [1, 0.5, 0.4, 0.5, 0.2, 0.3],
     [1, 0.4, 0.5, 0.3, 0.1, 0.9],
     [1, 0.2, 0.5, 0.8, 0.8, 0.7],
     [1, 0.6, 0.4, 0.6, 0.5, 0.9]]

b = []

for item in a:
    b.append(item[1])

# print(max(b))

# pred_boxes = [class, prob, x1, x2, y1, y2]

def _nms (pred_boxes, prob_threshold, iou_threshold, box_format):

    assert type(pred_boxes) == list

    # step 1: filtering bboxes with prob < threshold

    bboxes = []
    for box in pred_boxes:
        if box[1] > prob_threshold:
            bboxes.append(box)

    bboxes.sort(key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:

        chosen_box = bboxes.pop(0)

        for box in bboxes:
            pred_box = torch.tensor(chosen_box[2:])
            target_box = torch.tensor(box[2:])

        # if box of different class --> kept
            if box[0] != chosen_box[0]:
                continue
            elif intersection_over_union(pred_box, target_box, box_format) >= iou_threshold:
                bboxes.remove(box)
            else: continue


        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

    # return bboxes

# print(_nms(a, 0.3, 0.5, 0.1))

t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

t2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [2, 0.9, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

t3_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [2, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]

t4_boxes = [
            [1, 0.9, 0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [1, 0.8, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],
        ]


result = _nms(
            t4_boxes,
            prob_threshold=0.2,
            iou_threshold=7 / 20, # 0.35
            box_format="midpoint",
        )

result_2 = nms(
            t4_boxes,
            threshold=0.2,
            iou_threshold=7 / 20, # 0.35
            box_format="midpoint",
        )

print(result)
print(result_2)


a = torch.tensor(t1_boxes[0][2:])
b = torch.tensor(t1_boxes[1][2:])
# print(a)
# print(b)

# test = intersection_over_union(a,b,box_format="midpoint")

# print(test)





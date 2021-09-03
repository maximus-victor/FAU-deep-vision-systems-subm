import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, num_classes, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    metrix = {
    'epoch_val_loss': None,
    'timings': [],
    'frames': []
    }

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    epoch_val_loss = 0.
    total_class_acc = {(k): {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for k in range(num_classes)}
    total_acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    ov_acc = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}


    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # record a lot of times
        model_time = time.time()

        starter.record()
        outputs = model(images)
        ender.record()
        
        
        # ______________________________________________________________________________________
        
        # ADDED BY INCEPTIONEERS
        # Get Class Acc by comparing predicted classes in outputs with classes in targets
        # outputs['labels'] == targets['labels'] maybe?
        
        # add the count of correct prediction of the batch to a correct prediction list
        # correct_label += torch.sum(pred_t==y_batch_val).item() -->pred_t == outputs['labels'] y_batch_val == targets['labels']
        # how does comparing work here? are both dicts sorted the same???
        
        # add the count of all predictions of the batch to a total prediction list
        # total_label += len(targets['labels']) --> maybe? or maybe wrong code
        
        # calculate accuracy of all predicted samples so far
        # acc_val = 100 * correct_label/total_label
        # might be completely inaccurate depending on number of "good" class scores --> not sure how they play a role

        # Per Image
        #   Look at predicted Classes 
        #   predictions = set(output['labels'].tolist())
        #   num_preds = len(predictions)) # .size()[0]

        class_acc = {(k): {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for k in range(num_classes)} # ignore background class
        for target, output in zip(targets, outputs):
            # e.g 
            # 3 predictions {1, 2, 3}, 
            # 4 labels {2, 3, 4, 5}
            # TP 2 = {2, 3} / num_classes
            # TN = 1 - (TP + FP + FN)
            # FP 1 = {1} / num_classes
            # FN 2 = {4, 5} / num_classes
            labels = set(target['labels'].tolist())
            predictions = set()

            prediction_list = output['labels'].tolist()
            print("Predictions:", prediction_list)
            score_list = output['scores'].tolist()

            for idx, score in enumerate(output['scores'].tolist()):
                if score >= .8:
                    predictions.add(prediction_list[idx])

            # labels = set(target['labels'].tolist())

            # num_preds = len(predictions)
            for k, _ in class_acc.items():
                if k in predictions and k in labels:
                    class_acc[k]['TP'] += 1/(num_classes)
                elif k in predictions and k not in labels:
                    class_acc[k]['FP'] += 1/(num_classes)
                elif k not in predictions and k in labels:
                    class_acc[k]['FN'] += 1/(num_classes)
                else: 
                    class_acc[k]['TN'] += 1/(num_classes)

            # Accuracy v2
            # 0 = background
            # 3 predictions {1, 2, 3}
            # 4 labels {2, 3, 4, 5}
            inter = labels.intersection(predictions)
            print("Labels:", labels)
            print("Predictsions after scores:", predictions)
            print("Intersection:", inter)

            if len(inter - set([0])) > 0: # inter not empty e.g. {2,3}
                # TP
                ov_acc['TP'] +=1
            else:
                if labels == set([0]) and predictions == set([0]):
                    # TN
                    ov_acc['TN'] += 1
                elif predictions == set([0]): # predicition nur Background, obwohl labels nicht nur Backgroud
                    # FN
                    ov_acc['FN'] +=1
                else: # i.e. keine Überschneidung, kein Background in label oder prediction
                    # FP
                    ov_acc['FP'] +=1






        for k1 in total_class_acc.keys():
            total_class_acc[k1] = {k2: total_class_acc[k1][k2] + class_acc[k1][k2] for k2 in total_class_acc[k1].keys()}
            total_acc['TP'] += total_class_acc[k1]['TP']
            total_acc['TN'] += total_class_acc[k1]['TN']
            total_acc['FP'] += total_class_acc[k1]['FP']
            total_acc['FN'] += total_class_acc[k1]['FN']

        # _______________________________________________________________________________________

        # record inference time
        if device == torch.device('cuda'):
            torch.cuda.synchronize() 						

        laps_time = starter.elapsed_time(ender)
            
        metrix['timings'].append(laps_time / len(images))
        metrix['frames'].append(len(images) /laps_time )

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # ADDED BY INCEPTIONEERS
        # CALCULATE VAL LOSS: val loss is calculated in model.train() since there is no better way of obtaining the loss on the validation set
        # Loss is not returned in evaluation mode of Faster RCNN
        # this strategy may alter the numerical value but is valid in comparison to the train loss. See discussion: https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333/2
        model.train()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        #print("losses:", losses)

        # reduce losses over all GPUs for logging purposes -> Same in training
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        #print("loss_value:", loss_value)

        epoch_val_loss += loss_value * len(images)
        #print("Epoch_val_loss:", epoch_val_loss)
        model.eval()

    for k1 in total_class_acc.keys():
        TP = total_class_acc[k1]['TP']
        TN = total_class_acc[k1]['TN']
        FP = total_class_acc[k1]['FP']
        FN = total_class_acc[k1]['FN']
        print(f'Per-class accuracy (Class: {k1}): {(TP+TN)/(TP+TN+FP+FN)}')

    total = (total_acc['TP'] + total_acc['TN']) / (total_acc['TP'] + total_acc['TN'] + total_acc['FP'] + total_acc['FN'])
    print(f'Total Validation accuracy (distributed): {total}')

    over_acc = (ov_acc['TP'] + ov_acc['TN']) / (ov_acc['TP'] + ov_acc['TN'] + ov_acc['FP'] + ov_acc['FN'])
    print(f'Total Validation accuracy ("winner takes it all"): {over_acc}')

    metrix['epoch_val_loss'] = epoch_val_loss / len(data_loader)
    print("Epoch_val_loss:", epoch_val_loss / len(data_loader))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, metrix



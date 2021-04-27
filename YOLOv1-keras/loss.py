import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np

def cls_iou(bbox1, bbox2):
    """
    bbox1,2 => b * 7 * 7 * 4(x,y,h,w)
    """
    xmin = K.minimum((bbox1[:,:,:, 0] - bbox1[:,:,:, 2]/2), (bbox2[:,:,:, 0] - bbox2[:,:,:, 2]/2))
    xmax = K.maximum((bbox1[:,:,:, 0] + bbox1[:,:,:, 2]/2), (bbox2[:,:,:, 0] + bbox2[:,:,:, 2]/2))
    ymin = K.minimum((bbox1[:,:,:, 1] - bbox1[:,:,:, 3]/2), (bbox2[:,:,:, 1] - bbox2[:,:,:, 3]/2))
    ymax = K.maximum((bbox1[:,:,:, 1] + bbox1[:,:,:, 3]/2), (bbox2[:,:,:, 1] + bbox2[:,:,:, 3]/2))
    
    inter = (xmax - xmin) * (ymax - ymin) # b*7*7
    area1 = bbox1[:,:,:, 2] * bbox1[:,:,:, 3] # b*7*7
    area2 = bbox2[:,:,:, 2] * bbox2[:,:,:, 3] # b*7*7
    
    union = area1[:,:,:] + area2[:,:,:] - inter[:,:,:]
    iou = inter / union # b*7*7
    iou = K.expand_dims(iou)
    print(iou.shape)
    print(iou[0,0,:,:])
    return iou

def rg_bbox_coord_loss(true, pred):
    lambda_coord=5
    lambda_noobj=0.5
    
    true_class = true[..., 10:] # b * 7 * 7 * 20
    true_box = true[..., :4]    # b * 7 * 7 * 4 (0:5 == 5:10)
    true_conf = K.expand_dims(true[..., 4])   # b*7*7*1
    
    pred_class = pred[..., 10:] # b*7*7*20
    pred_box1 = pred[..., :4]    # b*7*7*4
    pred_box2 = pred[..., 5:9]   # b*7*7*4
    pred_conf1 = K.expand_dims(pred[..., 4])    # b*7*7*1
    pred_conf2 = K.expand_dims(pred[..., 9])    # b*7*7*1
    pred_confs = K.concatenate([pred_conf1, pred_conf2]) # b*7*7*2
    
    # cls IoU and select best box iou
    box1_iou = cls_iou(true_box, pred_box1) # b*7*7*1
    box2_iou = cls_iou(true_box, pred_box2) # b*7*7*1
    
    box_ious = K.concatenate([box1_iou, box2_iou], axis=3)  # b*7*7*2
    box_ious = K.expand_dims(box_ious)      # b*7*7*2*1
    
    best_iou = K.max(box_ious, axis=4)  # b*7*7*2
    best_p = K.max(best_iou, axis=3, keepdims=True)  # b*7*7*1

    box_p = K.cast(best_iou >= best_p, K.dtype(best_p)) # b*7*7*2

    noobj_loss = lambda_noobj * (1 - box_p*true_conf) * K.square(0 - pred_confs) # b*7*7*2
    obj_loss = box_p * true_conf * K.square(1-pred_confs) # b*7*7*2

    # Confidence Loss
    conf_loss = K.sum(noobj_loss + obj_loss)
    
    # Class Loss
    class_loss = true_conf * K.square(true_class - pred_class) # b*7*7*20
    class_loss = K.sum(class_loss)
    
    # Box Loss
    pred_box_xy = K.concatenate([K.expand_dims(pred_box1[...,:2], axis=3), 
                                 K.expand_dims(pred_box2[...,:2], axis=3)], 
                                 axis=3) # b*7*7*2*2
    pred_box_wh = K.concatenate([K.expand_dims(pred_box1[...,2:], axis=3), 
                                 K.expand_dims(pred_box2[...,2:], axis=3)], 
                                 axis=3) # b*7*7*2*2
    true_box_xy = K.expand_dims(true_box[...,:2], axis=3) # b*7*7*1*2
    true_box_wh = K.expand_dims(true_box[...,2:], axis=3) # b*7*7*1*2
    box_p = K.expand_dims(box_p)     # b*7*7*1*1
    true_conf = K.expand_dims(true_conf)     # b*7*7*1*1
    
    coord_loss = lambda_coord * box_p * true_conf * K.square((true_box_xy - pred_box_xy))  # b*7*7*2*2
    line_loss = lambda_coord * box_p * true_conf * K.square((K.sqrt(true_box_wh) - K.sqrt(pred_box_wh))) # b*7*7*2*2
    box_loss = K.sum(coord_loss + line_loss)
    
    loss = box_loss + conf_loss + class_loss

    return loss


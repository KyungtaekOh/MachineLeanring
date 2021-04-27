import tensorflow as tf

def cls_iou(bbox1, bbox2):
    """
    TODO:
    bbox1,2 => b * 7 * 7 * 4(x,y,h,w)
    """
    # shape => b*7*7
    print(bbox1.shape, bbox2.shape)
    xmin = K.min((bbox1[..., 0] - bbox1[..., 2]/2), (bbox2[..., 0] - bbox2[..., 2]/2))
    xmax = K.max((bbox1[..., 0] + bbox1[..., 2]/2), (bbox2[..., 0] + bbox2[..., 2]/2))
    ymin = K.min((bbox1[..., 1] - bbox1[..., 3]/2), (bbox2[..., 1] - bbox2[..., 3]/2))
    ymax = K.max((bbox1[..., 1] + bbox1[..., 3]/2), (bbox2[..., 1] + bbox2[..., 3]/2))
    
    inter = (xmax - xmin) * (ymax - ymin) # b*7*7
    area1 = bbox1[..., 2] * bbox1[..., 3] # b*7*7
    area2 = bbox2[..., 2] * bbox2[..., 3] # b*7*7
    
    union = area1[:,:,:] + area2[:,:,:] - inter[:,:,:]
    iou = inter / union # b*7*7
    iou = K.expand_dims(iou)
    return iou

def rg_bbox_coord_loss(true, pred):
    S=7
    B=2
    C=20
    N=5*B + C
    lambda_coord=5
    lambda_noobj=0.5
    batch_size = pred.shape[0]
    
    true_class = true[..., 10:]  # b * 7 * 7 * 20
    true_box = true[..., :4]  #  b * 7 * 7 * 4 (0:5 == 5:10)
    true_conf = true[..., :4] # b*7*7
    
    pred_class1 = pred[..., 10:] # b*7*7*20
    pred_box1 = pred[..., :4]    # b*7*7*4
    pred_box2 = pred[..., 5:9]   # b*7*7*4
    pred_conf1 = pred[..., 4]    # b*7*7
    pred_conf2 = pred[..., 9]    # b*7*7
    
    # cls IoU and select best box iou
    box1_iou = cls_iou(true_box, pred_box1) # b*7*7
#     box2_iou = cls_iou(true_box, pred_box2) # b*7*7
#     box_ious = K.concatenate([box1_iou, box2_iou], )
    return true_box, pred_box2


if __name__=="__main__":
    n1 = np.zeros((16,7,7,30))
    n1[:,:,:,:10] = [1,2,3,4,1,2,3,4,5,1]
    n1[:,:,:,:15] = 1
    n2 = np.zeros((16,7,7,30))
    n2[:,:,:,:10] = [1,2,3,4,1,2,3,4,5,1]
    n2[:,:,:,:15] = 1
    # print(n1.shape, n2.shape)
    b1, b2 = rg_bbox_coord_loss(n1, n2)
    print(b1.shape, b2.shape)


# class Loss(tf.keras.losses.Loss):
#     def __init__(grid=7, num_bboxes=2, num_classes=20):
#         super().__init__()
#         self.coord_value = 5
#         self.noob_value = 0.5
#         self.boxes = []
#         self.S = grid
#         self.B = num_bboxes
#         self.C = num_classes
#         self.N = 5*self.B + self.C
#         self.lambda_coord = 5
#         self.lambda_noobj = 0.5
#     """
#     pred_shape = (batch, grid, grid, 30)
#     grid (7, 7)
#         [0:5]   = (box1) x, y, w, h, p(obj)
#         [5:10]  = (box2) x, y, w, h, p(obj)
#         [10:30] = Class(VOC have 20 class)
#     """
#     def reg_bbox(self, true, pred):
#         true_c = true[:, :, :, :]
#         pred_C = pred[:, :, :, :]
        



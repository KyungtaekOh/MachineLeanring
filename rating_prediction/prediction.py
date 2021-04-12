import os
import model_resnet
import tools
from collections import Counter

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model = model_resnet.getResnet50()
weight = r'/home/cvlab09/kyung-taek/cnn/saved_weight/Resent50_epoch500_batch16_trial1.h5'
model.load_weights(weight)


directory = r'/home/cvlab09/kyung-taek/cnn/'
img_dir = 'data/train/'
file_dir = 'splited_valid.txt'
pred = tools.Prediction(directory, img_dir, file_dir, 9)
correct, mae = pred.prediction(model)

result = pred.pred_class
print(result)
print("[Correct / Total] :", correct, "  MAE :", mae.numpy())

# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import torch
import time
from PIL import Image

from torch.autograd import Variable
from dataset import get_cityscapes_loader
from iouEval import iouEval, getColorEntry

# verificare come utilizzare il parametro method

def main(method, model, datadir, cpu, num_classes):

    # load the dataset
    loader = get_cityscapes_loader(datadir)

    # create the IoU evaluator
    iouEvalVal = iouEval(num_classes)

    # start the timer used for the prints
    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):

        # if the cpu flag is not set, move the data to the gpu
        if (not cpu):
            images = images.cuda()
            labels = labels.cuda()

        # launch the model with the images as input while disabling gradient computation
        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        # anomaly_scores = get_anomaly_score(outputs, method)

        outputs = outputs.max(1)[1].unsqueeze(1).data
        labels = labels.unsqueeze(0).data

        # add the batch to the IoU evaluator
        iouEvalVal.addBatch(outputs, labels)

        # print the filename of the image
        filenameSave = filename[0].split("leftImg8bit/")[1] 
        print (step, filenameSave)

    # get the IoU results
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []

    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

    # apre il file in modalità append (aggiunge testo alla fine)
    file = open('results.txt', 'a')

    file.write(method + "\tMEAN IoU: ", iouStr, "%")
    file.write("\n")

    # chiude il file
    file.close()
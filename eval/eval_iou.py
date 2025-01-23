# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

from print_output import print_output
import torch
import time
from PIL import Image

from torch.autograd import Variable
from dataset import get_cityscapes_loader
from iouEval import iouEval, getColorEntry

# verificare come utilizzare il parametro method

def main(model, datadir, cpu, num_classes, ignoreIndex=19):

    # load the dataset
    loader = get_cityscapes_loader(datadir, 10, 'val')

    # create the IoU evaluator
    iouEvalVal = iouEval(num_classes, ignoreIndex=ignoreIndex)

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
            out = model(inputs)
            
        # get the max logit value for each pixel
        outputs = out.max(1)[1].unsqueeze(1).data
        labels = labels.unsqueeze(1).data

        # add the batch to the IoU evaluator
        iouEvalVal.addBatch(outputs, labels)

        # print the filename of the image
        filenameSave = filename[0].split("leftImg8bit/")[1] 
        print (step, filenameSave)

        if step in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
           print_output(out[0, :, :, :], filename[0].split("leftImg8bit/")[1])

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
    if ignoreIndex == -1:
        print(iou_classes_str[19], "void")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

    return iouVal
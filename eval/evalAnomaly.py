import os
import glob
import torch
from PIL import Image
import numpy as np
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
import torch.nn.functional as F


# *********************************************************************************************************************

def get_anomaly_score(result, method='MSP'):

    # filtering void class
    # result = result[:, :-1, :, :]
    
    if method == 'MSP':
        probabilities = F.softmax(result, dim=1)
        retval = 1 - np.max(probabilities.squeeze(0).data.cpu().numpy(), axis=0)
        return retval

    elif method == 'MaxEntropy':
        probabilities = F.softmax(result, dim=1)
        entropy = - np.sum(probabilities.squeeze(0).data.cpu().numpy() * np.log(probabilities.squeeze(0).data.cpu().numpy() + 1e-10), axis=0)
        return entropy

    elif method == 'MaxLogit':
        retval = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
        return retval
    
# ********************************************************************************************************************


def main(dataset_dir, model, method):

    # crea due liste vuote dove salvare i risultati
    ood_gts_list = []
    anomaly_score_list = []

    # for each path in the input path list (glob.glob returns a list of paths expanding the * wildcard)
    for path in glob.glob(dataset_dir):
        
        # print(path)

        # load the image, converting it to an RGB tensor (dimensions are W x H x 3)
        image = Image.open(path).convert('RGB')
        # converts the tensor to a numpy tensor and loads it on the gpu, adding a dimension and converting it to float (dimensions are 1 x H x W x 3)
        image = torch.from_numpy(np.array(image)).unsqueeze(0).float()
        # permutes the dimensions of the tensor (dimensions are 1 x 3 x H x W)
        image = image.permute(0,3,1,2)

        # launches the model with the image as input while disabling gradient computation (saves memory and computation time)
        with torch.no_grad():
            # result size is 1 x 20 x H x W
            # the model returns for each pixel the logits for each class
            result = model(image)
        
        # calculates the anomaly score using the method specified
        # anomaly_result size is H x W
        # the anomaly score is a measure of confident the model is about the prediction
        # a high anomaly score means the pixel might represent an object class out of the distribution
        anomaly_result = get_anomaly_score(result, method)
        
        # creates the path for the ground truth mask
        pathGT = path.replace("images", "labels_masks")
        
        # corrects the ground truth format if different from the images         
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")

        # opens the ground truth mask image and converts it to a numpy tensor
        mask = Image.open(pathGT)
        # ood_gts stands for out-of-distribution ground truths
        # the ground truth mask highlights the pixels that are not part of any class
        ood_gts = np.array(mask)

        # corrects the gray scale values of the ground truth mask (???)
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        # checks if the ground truth mask contains at least one pixel with value 1
        if 1 not in np.unique(ood_gts):
            continue
        else:
            # if the ground truth contains an anomaly, appends the ground truth mask and the anomaly score to the lists
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)

        # releases the memory used by the result, anomaly_result, ood_gts and mask tensors
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    print(f'Number of images: {len(ood_gts_list)}')

    # creates two numpy tensor from the lists
    ood_gts_np = np.array(ood_gts_list)
    anomaly_scores_np = np.array(anomaly_score_list)

    # creates two boolean lists of masks for the out-of-distribution and in-distribution ground truths
    ood_mask = (ood_gts_np == 1)
    ind_mask = (ood_gts_np == 0)
    
    # creates two lists filtering anomaly scores for the out-of-distribution and in-distribution ground truths
    ood_out = anomaly_scores_np[ood_mask]
    ind_out = anomaly_scores_np[ind_mask]

    # creates two lists of labes
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    # concatenates the lists of anomaly scores and labels
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    # the result is two lists, one for anomaly scores and the other for the labels indicating if the pixel is out-of-distribution or in-distribution
    # both lists are ordered by the label value

    print("Calculating AUPRC and FPR@TPR95...")

    # calculates the AUPRC score and the FPR@TPR95 score
    # both metrics work on anomaly scores and labels because they elaborate the right threshold and separate the two classes
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    return prc_auc, fpr


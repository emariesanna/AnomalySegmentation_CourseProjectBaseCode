import os
import random
import numpy as np
from state_dictionary import load_my_state_dict
import torch
from erfnet import ERFNet
from argparse import ArgumentParser
from temperature_scaling3 import ModelWithTemperature
from enet import ENet
from evalAnomaly import main as evalAnomaly
from eval_iou import main as eval_iou

# imposta il seed per il rng di python, numpy e pytorch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# forza pytorch a usare operazioni deterministiche
torch.backends.cudnn.deterministic = True
# abilita il benchmarking per ottimizzare le prestazioni, 
# a scapito di un lieve elemento di non-determinismo in alcuni scenari
torch.backends.cudnn.benchmark = True

# numero delle classi del dataset
NUM_CLASSES = 20
# flag per attivare valutazione di IOU
IOU = 1
# flag per attivare valutazione di Anomaly Detection tramite anomaly scores
ANOMALY = 0
# flag per attivare valutazione di Anomaly Detection tramite void class
VOID = 0
# modello da utilizzare (erfnet o enet)
MODEL = "enet"
# pesi prunati sì/no
PRUNED = 0
# flag per attivare la stampa di un certo numero di immagini
PRINT = 1

DatasetDir = {
    "LostFound": "./Dataset/Validation_Dataset/FS_LostFound_full/images/*.png",
    "FSstatic": "./Dataset/Validation_Dataset/fs_static/images/*.jpg",
    "RoadAnomaly": "./Dataset/Validation_Dataset/RoadAnomaly/images/*.jpg",
    "RoadAnomaly21": "./Dataset/Validation_Dataset/RoadAnomaly21/images/*.png",
    "RoadObsticle21": "./Dataset/Validation_Dataset/RoadObsticle21/images/*.webp",
              }

# *********************************************************************************************************************

def main():

    if MODEL == "erfnet":
        modelclass = "erfnet.py"
        if PRUNED == 0:
            weights = "erfnet_pretrained.pth"
        else:
            weights = "erfnetPruned.pth"
        Model = ERFNet
    elif MODEL == "enet":
        modelclass = "enet.py"
        weights = "checkpoint-epoch70-state-dict.pth"
        Model = ENet

    # definisce un parser, ovvero un oggetto che permette di leggere gli argomenti passati da riga di comando
    parser = ArgumentParser()
    # definisce gli argomenti accettati dal parser
    # nomi dei dataset da utilizzare
    parser.add_argument("--datasets",
                        default=["FSstatic","RoadAnomaly","RoadAnomaly21","RoadObsticle21","LostFound"],
                        nargs="+", help="A list of space separated dataset names")
    # directory per la cartella contentente il modello pre-addestrato
    parser.add_argument('--loadDir', default="./trained_models/")
    # file dei pesi (dentro la cartella loadDir)
    parser.add_argument('--loadWeights', default=weights)
    # directory per il modello
    parser.add_argument('--loadModel', default = modelclass)
    # cartella del dataset da utilizzare (val o train)
    parser.add_argument('--subset', default="val")
    # directory del dataset
    parser.add_argument('--datadir', default="./Dataset/Cityscapes")
    # numero di thread da usare per il caricamento dei dati
    parser.add_argument('--num-workers', type=int, default=4)
    # dimensione del batch per l'elaborazione delle immagini 
    # (quante immagini alla volta vengono elaborate, maggiore è più veloce ma richiede più memoria)
    parser.add_argument('--batch-size', type=int, default=20)
    # flag per forzare l'utilizzo della cpu (action='store_true' rende l'argomento opzionale e false di default)
    parser.add_argument('--cpu', action='store_true')
    # quale metodo utilizzare per l'anomaly detection
    parser.add_argument('--methods', default=["MaxLogit", "MaxEntropy", "MSP"],
                        nargs="+", help="A list of space separated method names between MSP, MaxEntropy and MaxLogit")
    # quale temperatura utilizzare per il temperature scaling
    parser.add_argument('--temperatures', default=[0, 0.5, 0.75, 1.1] , 
                        nargs="+", help="Set 0 to disable temperature scaling, set n to use learned temperature")	
    # costruisce un oggetto contenente gli argomenti passati da riga di comando (tipo Namespace)
    args = parser.parse_args()
    # inizializza due liste vuote per contenere i risultati

    # mette insieme gli argomenti del parser e definisce il path del modello e dei pesi
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # crea un modello ERFNet con NUM_CLASSES classi
    model = Model(NUM_CLASSES)

    # se l'argomento cpu non è stato passato, allora imposta torch per usare la gpu
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # crea uno state dictionary a partire dai pesi salvati
    # lo state dictionary è una struttura dati che contiene i pesi e i buffer del modello
    # (i buffer sono valori statici necessari per i calcoli, come ad esempio la media e la varianza)
    # la parte map_location serve a salvare il dizionario su un dispositivo diverso da quello in cui sono salvati i pesi
    state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)

    # carica nel modello lo state dictionary creato
    model = load_my_state_dict(model, state_dict)
    #model.load_state_dict(state_dict)

    print ("Model and weights LOADED successfully")

    # imposta il modello in modalità di valutazione
    # questo cambia alcuni comportamenti come la batch normalization 
    # (che viene calcolata su media e varianza globali invece che del batch) 
    # e il dropout (che viene disattivato)
    model.eval()

    file = open('results.txt', 'a')
    file.write("MODEL " + MODEL.capitalize() + "\n")
    file.close()

    # se non esiste il file results.txt, crea un file vuoto
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()

    if IOU == 1:
        print("Evaluating IOU")
        iou = eval_iou(model, args.datadir, cpu=False, num_classes=NUM_CLASSES, ignoreIndex=19, model_name=MODEL, PRINT=PRINT)
        file = open('results.txt', 'a')
        file.write("MEAN IoU: " + '{:0.2f}'.format(iou*100) + "%")
        file.write("\n")
        file.close()
    
    if ANOMALY == 1:
        print("Evaluating Anomaly Detection")

        def iterate_datasets(mod):
            for dataset in args.datasets:
                dataset_string = "Dataset " + dataset
                dataset_dir = DatasetDir[dataset]
                prc_auc, fpr = evalAnomaly(dataset_dir, mod, method)
                result_string = 'AUPRC score:' + str(prc_auc*100.0) + '\tFPR@TPR95:' + str(fpr*100.0)
                print(temperature_string + dataset_string + method_string + "\n" + result_string)
                file = open('results.txt', 'a')
                file.write(temperature_string + dataset_string + method_string + "\n" + result_string + "\n")
                file.close()

        for method in args.methods:

            method_string = " using method: " + method

            if method == "MSP":
                for temperature in args.temperatures:
                    if temperature != 0:
                        model_t = ModelWithTemperature(model, temperature)
                    else:
                        model_t = model
                    iterate_datasets(model_t)
                    temperature_string = "Temperature Scaling: " + str(temperature) + "\t"

            temperature_string = ""
            iterate_datasets(model)

    if VOID == 1:
        print("Evaluating Void Class IoU")
        iou = eval_iou(model, args.datadir, cpu=False, num_classes=NUM_CLASSES, ignoreIndex=-1, PRINT=PRINT)
        file = open('results.txt', 'a')
        file.write("MEAN IoU: " + '{:0.2f}'.format(iou*100) + "%")
        file.write("\n")
        file.close()

    file = open('results.txt', 'a')
    file.write("\n\n")
    file.close()
            
            
if __name__ == '__main__':
    main()
import os
import glob
import random
from PIL import Image
import numpy as np
import torch
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from temperature_scaling import ModelWithTemperature
from enet_pytorch_cityscapes import cityscapes_enet_pytorch as ENet
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
# flag per passare da valutazione di IOU a valutazione di Anomaly Detection
IOU = False
# modello da utilizzare (erfnet o enet)
MODEL = "erfnet"
DatasetDir = {
    "LostFound": "./Dataset/Validation_Dataset/FS_LostFound_full/images/*.png",
    "FSstatic": "./Dataset/Validation_Dataset/fs_static/images/*.jpg",
    "RoadAnomaly": "./Dataset/Validation_Dataset/RoadAnomaly/images/*.jpg",
    "RoadAnomaly21": "./Dataset/Validation_Dataset/RoadAnomaly21/images/*.png",
    "RoadObsticle21": "./Dataset/Validation_Dataset/RoadObsticle21/images/*.webp",
              }


# *********************************************************************************************************************

# funzione per copiare i pesi da uno state dictionary ad un modello
# gestisce anche i casi in cui state_dict ha nomi di parametri diversi da quelli attesi dal modello (own_state)
# in particolare, se i nomi dei parametri in state_dict hanno un prefisso "module." (come quando si salva un modello con DataParallel)
# allora viene rimosso il prefisso prima che il parametro venga copiato nel modello
def load_my_state_dict(model, state_dict):
        # recupera lo state dictionary attuale del modello
        own_state = model.state_dict()
        # per ogni parametro nello state dictionary passato alla funzione
        # (è un dizionario quindi fatto di coppie chiave-valore)
        for name, param in state_dict.items():
            # se il nome del parametro non è presente nello state dictionary del modello
            if name not in own_state:
                # se il nome del parametro ha il prefisso "module."
                if name.startswith("module."):
                    # rimuove il prefisso
                    name = name.split("module.")[-1]
                    # copia il parametro di state_dict nel modello
                    own_state[name].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            # se il nome del parametro è nel modello
            else:
                # copia il parametro di state_dict nel modello
                own_state[name].copy_(param)
        return model
    
# ********************************************************************************************************************


def main():

    if MODEL == "erfnet":
        modelclass = "erfnet.py"
        weights = "erfnet_pretrained.pth"
        Model = ERFNet
    elif MODEL == "enet":
        modelclass = "enet.py"
        weights = "enet_pytorch_cityscapes.pth"
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
    parser.add_argument('--batch-size', type=int, default=1)
    # flag per forzare l'utilizzo della cpu (action='store_true' rende l'argomento opzionale e false di default)
    parser.add_argument('--cpu', action='store_true')
    # quale metodo utilizzare per l'anomaly detection
    parser.add_argument('--method', default="MaxLogit", help="Can be MSP, MaxEntropy or MaxLogit")
    # quale temperatura utilizzare per il temperature scaling
    parser.add_argument('--temperature', default=0 , help="Set 0 to disable temperature scaling")
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

    print ("Model and weights LOADED successfully")

    if args.temperature != 0:
        model = ModelWithTemperature(model, args.temperature)

    # imposta il modello in modalità di valutazione
    # questo cambia alcuni comportamenti come la batch normalization 
    # (che viene calcolata su media e varianza globali invece che del batch) 
    # e il dropout (che viene disattivato)
    model.eval()

    if IOU:
        print("Evaluating IOU")
        eval_iou(args.datadir, args.cpu, NUM_CLASSES, model)
    else:
        print("Evaluating Anomaly Detection")
        for dataset in args.datasets:
            print("Dataset", dataset)
            dataset_dir = DatasetDir[dataset]
            evalAnomaly(dataset_dir, dataset , model, args.method)

if __name__ == '__main__':
    main()
#from eval import evalAnomaly
import argparse
from evalAnomaly import main as EAmain
from evalAnomalyTemperature import main as EATmain
import os

def main1():
    root = './Dataset/Validation_Dataset/'
    paths = ['RoadAnomaly21/images/*.png', 'RoadObsticle21/images/*.webp', 'RoadAnomaly/images/*.jpg', 'FS_LostFound_full/images/*.png', 'fs_static/images/*.jpg']
    methods = ['MSP', 'MaxEntropy','MaxLogit']
    for method in methods:

        if not os.path.exists('results.txt'):
            open('results.txt', 'w').close()
        file = open('results.txt', 'a')
        file.write( "\n"+ method + '\n')

        for path in paths:
            print("now executing: ", path, "With Method: ", method)
    
            EAmain(root + path, method)


def main2():
    root = './Dataset/Validation_Dataset/'
    paths = ['RoadAnomaly21/images/*.png', 'RoadObsticle21/images/*.webp', 'RoadAnomaly/images/*.jpg', 'FS_LostFound_full/images/*.png', 'fs_static/images/*.jpg']
    temperatures = [1.7066885232925415]
    if not os.path.exists('results_temp.txt'):
            open('results_temp.txt', 'w').close()
    file = open('results_temp.txt', 'a')
    for t in temperatures:

        
        file.write( "\n t ="+ str(t) + '\n')

        for path in paths:
            print("now executing: ", path, "With temperature: ", t)
    
            EATmain(root + path, t)
    file.close()

if __name__ == '__main__':
    main2()
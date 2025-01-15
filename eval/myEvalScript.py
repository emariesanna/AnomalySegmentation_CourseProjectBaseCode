#from eval import evalAnomaly
import argparse
from evalAnomaly import main as EAmain
import os

def main():
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

if __name__ == '__main__':
    main()
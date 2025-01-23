import torch
import numpy as np
import matplotlib.pyplot as plt

def print_output(output, filename):

    # Funzione per mappare le classi ai colori
    def create_color_map():
        return np.array([
            [255, 0, 0],    # Classe 0 - Rosso
            [0, 255, 0],    # Classe 1 - Verde
            [0, 0, 255],    # Classe 2 - Blu
            [255, 255, 0],  # Classe 3 - Giallo
            [255, 0, 255],  # Classe 4 - Magenta
            [0, 255, 255],  # Classe 5 - Ciano
            [128, 0, 0],    # Classe 6 - Marrone
            [0, 128, 0],    # Classe 7 - Verde scuro
            [0, 0, 128],    # Classe 8 - Blu scuro
            [128, 128, 0],  # Classe 9 - Oliva
            [128, 0, 128],  # Classe 10 - Viola
            [0, 128, 128],  # Classe 11 - Verde acqua scuro
            [192, 192, 192],# Classe 12 - Grigio chiaro
            [128, 128, 128],# Classe 13 - Grigio
            [64, 64, 64],   # Classe 14 - Grigio scuro
            [255, 128, 0],  # Classe 15 - Arancione
            [128, 64, 0],   # Classe 16 - Bronzo
            [64, 128, 0],   # Classe 17 - Verde oliva
            [0, 64, 128],   # Classe 18 - Blu petrolio
            [0, 0, 0],      # Classe 19 - Nero
        ], dtype=np.uint8)

    # Simulazione di output del modello (logit 20x512x1024)
    # Usa torch.rand per generare valori casuali come esempio
    logits = output

    # Trova la classe con il logit massimo per ogni pixel
    predicted_classes = torch.argmax(logits, dim=0).cpu().numpy()

    # Crea la mappa dei colori
    color_map = create_color_map()

    # Crea l'immagine RGB a partire dalle classi previste
    height, width = predicted_classes.shape
    segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)
    for cls in range(20):
        segmentation_image[predicted_classes == cls] = color_map[cls]

    # Mostra l'immagine risultante
    plt.figure(figsize=(12, 6))
    plt.imshow(segmentation_image)
    plt.axis('off')
    plt.title(filename)
    plt.show()

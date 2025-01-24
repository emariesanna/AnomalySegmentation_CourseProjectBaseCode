import torch

# Carica il file .pth
file_path = "./trained_models/enet_best_model.pth"
data = torch.load(file_path)

# Accedi al dizionario desiderato
if "state_dict" in data:
    state_dict = data["state_dict"]
else:
    raise KeyError("Il dizionario 'state_dict' non è presente nel file!")

# Salva solo il dizionario estratto in un nuovo file .pth
new_file_path = "enet_best_model_state_dict.pth"
torch.save(state_dict, new_file_path)

print(f"Dizionario salvato in {new_file_path}")
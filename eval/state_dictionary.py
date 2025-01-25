import torch


def extract_state_dictionary():

    # Carica il file .pth
    file_path = "./trained_models/checkpoint-epoch70.pth"
    data = torch.load(file_path)

    # Accedi al dizionario desiderato
    if "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        raise KeyError("Il dizionario 'state_dict' non è presente nel file!")

    # Salva solo il dizionario estratto in un nuovo file .pth
    new_file_path = "./trained_models/checkpoint-epoch70-state-dict.pth"
    torch.save(state_dict, new_file_path)

    print(f"Dizionario salvato in {new_file_path}")


# funzione per copiare i pesi da uno state dictionary ad un modello
# gestisce anche i casi in cui state_dict ha nomi di parametri diversi da quelli attesi dal modello (own_state)
# in particolare, se i nomi dei parametri in state_dict hanno un prefisso "module." (come quando si salva un modello con DataParallel)
# allora viene rimosso il prefisso prima che il parametro venga copiato nel modello
def load_my_state_dict(model, state_dict):
        # recupera lo state dictionary attuale del modello
        own_state = model.state_dict()

        open('keys.txt', 'w').close()

        file = open('keys.txt', 'a')

        file.write("Model state dict size: " + str(len(own_state.keys())))
        file.write("\n")
        file.write("Uploaded state dict size: " + str(len(state_dict.keys())))
        file.write("\n")
        for step in range(0, max(len(own_state.keys()), len(state_dict.keys()))):
            if step < len(own_state.keys()):
                own_str = str(list(own_state.keys())[step])
            else:
                own_str = ""
            if step < len(state_dict.keys()):
                state_str = str(list(state_dict.keys())[step])
            else:
                state_str = ""
            file.write(str(step) + "\t" + own_str + "\t" + state_str + "\n") 
        
        not_loaded = []
        missing = []

        for name in own_state:
            found = False
            for name2 in state_dict:
                if name == name2 or name == ("module." + name2) or ("module." + name) == name2:
                    found = True
                else:
                    pass
            if not found:
                missing.append(name)

        # per ogni parametro nello state dictionary passato alla funzione
        # (è un dizionario quindi fatto di coppie chiave-valore)
        for name, param in state_dict.items():
            loaded = False
            for name2 in own_state:
                if name == name2:
                    print(name, name2)
                    print(param.size(), own_state[name2].size())
                    print("\n")
                    own_state[name].copy_(param)
                    loaded = True
                elif name == ("module." + name2):
                    own_state[name.split("module.")[-1]].copy_(param)
                    loaded = True
                elif ("module." + name) == name2:
                    own_state[("module." + name)].copy_(param)
                    loaded = True
                else:
                    pass
            if not loaded:
                print(name, " not loaded")
                not_loaded.append(name)

        file.write("\n")
        file.write("Not loaded: " + str(len(not_loaded)))
        file.write("\n")
        for step in range(0, len(not_loaded)):
            file.write(str(step) + "\t" + not_loaded[step] + "\n")
        file.write("\n")
        file.write("Missing: " + str(len(missing)))
        file.write("\n")
        for step in range(0, len(missing)):
            file.write(str(step) + "\t" + missing[step] + "\n")
        file.write("\n")

        file.close()

        return model


if __name__ == "__main__":
    extract_state_dictionary()
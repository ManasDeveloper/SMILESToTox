import torch
import pickle
from lstm_model import SMILESLstm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocab
with open("./vocab.pkl", "rb") as f:
    char_to_idx = pickle.load(f)

# Load model
vocab_size = len(char_to_idx)
model = SMILESLstm(vocab_size)
model.load_state_dict(torch.load("lstm_model_target_prediction.pth", map_location=device))
model.to(device)
model.eval()

# Target labels
tox21_targets = [
    ("NR-AR", "Androgen Receptor"),
    ("NR-AR-LBD", "Androgen Receptor LBD"),
    ("NR-AhR", "Aryl Hydrocarbon Receptor"),
    ("NR-Aromatase", "Aromatase"),
    ("NR-ER", "Estrogen Receptor"),
    ("NR-ER-LBD", "Estrogen Receptor LBD"),
    ("NR-PPAR-gamma", "PPAR Gamma Receptor"),
    ("SR-ARE", "Antioxidant Response Element"),
    ("SR-ATAD5", "ATAD5 DNA Repair Pathway"),
    ("SR-HSE", "Heat Shock Response Element"),
    ("SR-MMP", "Mitochondrial Membrane Potential"),
    ("SR-p53", "p53 Tumor Suppressor Pathway")
]

def encode_smiles(smiles, max_len=120):
    pad_idx = char_to_idx['<PAD>']
    encoded = [char_to_idx[c] if c in char_to_idx else pad_idx for c in smiles]
    if len(encoded) < max_len:
        encoded += [pad_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return torch.tensor(encoded).unsqueeze(0).to(device)

@torch.no_grad()
def predict(smiles: str, threshold: float = 0.5):
    x = encode_smiles(smiles)
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    preds = (probs > threshold).astype(int)

    results = []
    for i, (code, name) in enumerate(tox21_targets):
        results.append({
            "target": name,
            "score": round(float(probs[i]), 3),
            "is_toxic": bool(preds[i])
        })

    return results

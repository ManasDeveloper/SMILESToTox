# 🧪 SMILES2Tox

SMILES2Tox is a deep learning-based toxicity prediction system that takes a molecule's SMILES representation as input and predicts its toxic effects on multiple biological targets using a BiLSTM architecture.

This project is trained on the **Tox21 dataset** — a benchmark dataset of 12 toxicity-related biological pathways — and supports real-time predictions via a **Streamlit-based web interface**.

---

## 🚀 Features

- 🔬 Predicts toxicity across **12 biological targets** (e.g., AR, ER, p53)
- 🔁 Built on a **bidirectional LSTM** trained from scratch
- 🧠 Handles character-level **SMILES embeddings**
- 📊 Visualizes toxicity scores with color-coded bar charts
- 🌐 Deployable as an interactive **Streamlit app**

---

## 📚 Technologies Used

- PyTorch (for model architecture and training)
- Streamlit (for interactive web UI)
- Tox21 dataset (multi-label toxicity classification)
- Matplotlib (for bar chart visualization)
- Python 🐍

---

## 📸 Sample Output

| Target            | Score | Toxic |
| ----------------- | ----- | ----- |
| Androgen Receptor | 0.742 | ✅ Yes |
| Estrogen Receptor | 0.183 | ❌ No  |
| ...               |       |       |



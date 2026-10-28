# Addestramento del lettore di cifre

Il modello (`../model.bin`) legge una cella scritta a mano (altezza 32 px) e restituisce le cifre (CTC).

- `seqtrain.py` — addestramento da zero su sequenze sintetiche costruite con MNIST:
  `python3 seqtrain.py mnist.npz seq.pt 240000 14`
- `finetune.py` — riaddestramento con le correzioni esportate dall'app (sezione "Migliora la lettura"):
  `python3 finetune.py mnist.npz seq.pt seq-new.pt ricontrolli-correzioni-*.json`
- `export.py` — pesi in formato per l'app: `python3 export.py seq-new.pt ../model.bin`

Dopo aver sostituito `model.bin`, aumenta `MODEL_VERSION` in `index.html` (così i telefoni scaricano il modello nuovo).

`mnist.npz`: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz (non incluso nel repo).

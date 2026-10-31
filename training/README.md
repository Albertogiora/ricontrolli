# Addestramento del lettore di cifre

Il modello (`../model.bin`) legge una cella scritta a mano (altezza 32 px) e restituisce le cifre (CTC).

- `seqtrain.py` — addestramento da zero su sequenze sintetiche costruite con MNIST:
  `python3 seqtrain.py mnist.npz seq.pt 240000 14`
- `finetune.py` — riaddestramento con le correzioni esportate dall'app (sezione "Migliora la lettura"):
  `python3 finetune.py mnist.npz seq.pt seq-new.pt ricontrolli-correzioni-*.json`
- `export.py` — pesi in formato per l'app: `python3 export.py seq-new.pt ../model.bin`
- `finetune_jax.py` — in alternativa, riaddestramento senza PyTorch partendo direttamente da `model.bin` (JAX su CPU):
  `pip install jax optax opencv-python-headless numpy` e poi
  `python3 finetune_jax.py ../model.bin nuovo.bin mnist.npz ricontrolli-correzioni-*.json`
  (`FROM_T=<ms>` ignora i campioni non controllati più vecchi di quella data; `HOLD=<src>` tiene fuori dei campioni per misurare). Il file `nuovo.bin` va copiato al posto di `../model.bin`.

I file con le correzioni contengono immagini dei fogli: non vanno messi nel repository (è pubblico).

Dopo aver sostituito `model.bin`, aumenta `MODEL_VERSION` in `index.html` (così i telefoni scaricano il modello nuovo).

`mnist.npz`: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz (non incluso nel repo).

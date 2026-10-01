"""Assignment 2 - Learning Strategy (MNIST)
Nur Ilham Iskandar - NIM 572104 - Magister Kecerdasan Artifisial UGM

Menjalankan replikasi slide (Bagian 1) dan 5 skenario eksperimen (Bagian 2).
Output: grafik loss/accuracy dan results.json di folder figures/.
"""
import os, json, io, contextlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Input, Flatten, Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
results = {}

# ======================= PART 1: replicate slides 14-23 =======================
keras.utils.set_random_seed(42)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model1 = Sequential([Input(shape=(28, 28, 1)), Flatten(),
                     Dense(64, activation="relu"), Dense(10, activation="softmax")])
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
history1 = model1.fit(X_train, y_train, epochs=10, batch_size=100,
                      validation_data=(X_test, y_test), verbose=0)
model1.save(f"{OUT}/my_model1.h5")
l, a = model1.evaluate(X_test, y_test, verbose=0)
pred = load_model(f"{OUT}/my_model1.h5").predict(X_test, verbose=0)
results["replica"] = dict(test_loss=l, test_acc=a, params=model1.count_params(),
                          actual30=int(np.argmax(y_test[30])), pred30=int(np.argmax(pred[30])),
                          loss=history1.history["loss"], val_loss=history1.history["val_loss"])
plt.figure(figsize=(6, 4))
plt.plot(range(10), history1.history["loss"], "r", label="training loss ANN")
plt.plot(range(10), history1.history["val_loss"], "b", label="validation loss ANN")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout()
plt.savefig(f"{OUT}/replica_loss.png", dpi=150); plt.close()
print("replica", l, a)

# ======================= PART 2: experiments =======================
# 55,000 train / 5,000 validation / 10,000 test (as on slide 18) so that the test set
# is never used for model selection (important for early stopping).
X_tr, Y_tr = X_train[:55000], y_train[:55000]
X_val, Y_val = X_train[55000:], y_train[55000:]
EPOCHS, BATCH = 30, 100   # longer than 10 so overfitting becomes visible

L1_RATE, L2_RATE, DROP_RATE = 1e-4, 1e-3, 0.3

def build(l1=None, l2=None, dropout=None):
    reg = regularizers.l1(l1) if l1 else (regularizers.l2(l2) if l2 else None)
    layers = [Input(shape=(28, 28, 1)), Flatten(),
              Dense(64, activation="relu", kernel_regularizer=reg)]
    if dropout: layers.append(Dropout(dropout))
    layers.append(Dense(10, activation="softmax", kernel_regularizer=reg))
    m = Sequential(layers)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return m

scenarios = {
    "baseline": dict(),
    "2a_L1": dict(l1=L1_RATE),
    "2b_L2": dict(l2=L2_RATE),
    "2c_Dropout": dict(dropout=DROP_RATE),
    "2d_EarlyStopping": dict(es=True),
    "2e_L1_Dropout": dict(l1=L1_RATE, dropout=DROP_RATE),
}

for name, cfg in scenarios.items():
    keras.utils.set_random_seed(42)
    es = cfg.pop("es", False)
    m = build(**cfg)
    cb = [EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)] if es else []
    h = m.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH,
              validation_data=(X_val, Y_val), callbacks=cb, verbose=0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.summary(line_length=70)
    # "pure" cross-entropy on train/test (without the penalty term) for fair comparison
    ce = keras.losses.CategoricalCrossentropy()
    tr_pred = m.predict(X_tr, verbose=0, batch_size=1000)
    te_pred = m.predict(X_test, verbose=0, batch_size=1000)
    tr_acc = float((tr_pred.argmax(1) == Y_tr.argmax(1)).mean())
    te_acc = float((te_pred.argmax(1) == y_test.argmax(1)).mean())
    te_loss_reported, _ = m.evaluate(X_test, y_test, verbose=0)
    w = m.layers[1].get_weights()[0]  # first Dense kernel (784x64)
    hist = h.history
    best_ep = int(np.argmin(hist["val_loss"])) + 1
    results[name] = dict(
        train_acc=tr_acc, test_acc=te_acc,
        train_ce=float(ce(Y_tr, tr_pred)), test_ce=float(ce(y_test, te_pred)),
        test_loss_reported=float(te_loss_reported),
        gap=tr_acc - te_acc, epochs_run=len(hist["loss"]), best_epoch=best_ep,
        min_val_loss=float(min(hist["val_loss"])),
        near_zero_w=float((np.abs(w) < 1e-3).mean()), mean_abs_w=float(np.abs(w).mean()),
        params=m.count_params(), summary=buf.getvalue(),
        loss=hist["loss"], val_loss=hist["val_loss"], acc=hist["acc"], val_acc=hist["val_acc"])
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name].items()
                 if k not in ("summary", "loss", "val_loss", "acc", "val_acc")})

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = range(1, len(hist["loss"]) + 1)
    ax[0].plot(ep, hist["loss"], "r", label="training loss")
    ax[0].plot(ep, hist["val_loss"], "b", label="validation loss")
    if es: ax[0].axvline(best_ep, ls="--", c="gray", label=f"best epoch ({best_ep})")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["acc"], "r", label="training acc")
    ax[1].plot(ep, hist["val_acc"], "b", label="validation acc")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=150); plt.close(fig)

json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
print("done")

# Deep Facial Recognition (Kivy Computer Vision App) with OpenCV and TensorFlow

A learning project that builds a facial verification system with a Siamese neural network and wraps it in a Kivy desktop app for real-time verification from a webcam.

## What it does

- Collects anchor (reference) and positive images of a person from a webcam, plus negative images of other people
- Trains a Siamese network to decide whether two face images show the same person
- Verifies a live webcam face against stored reference images in a Kivy GUI app

## Dataset

Self-collected webcam images under `data/`, in three folders:

- `data/anchor/` — reference images of the person to verify
- `data/positive/` — more images of the same person
- `data/negative/` — images of other people

Images are loaded, resized to 100x100, and split into train/test partitions of anchor-positive and anchor-negative pairs.

## Model

A Siamese network built in Keras:

- **Embedding network:** convolutional blocks (64 and 128 filters) over 100x100x3 inputs, producing a 4096-dimensional embedding
- **Distance layer:** a custom `L1Dist` layer computing the absolute difference between the two embeddings
- **Output:** Dense(1, sigmoid) — the probability that the two images show the same person
- **Training:** binary cross-entropy loss, Adam optimizer (learning rate 1e-4), 50 epochs, with checkpoints saved during training

The trained model is saved as `siameseModel.h5`.

## Results

No aggregate accuracy metric is reported in the notebook. Evaluation runs the model on held-out verification pairs and prints a per-pair verification score (near 1 for same-person pairs, near 0 otherwise), plus a detection threshold and verification decision for the live webcam test.

## Project structure

```
Facial Verification.ipynb   # data collection, training, evaluation, real-time test
siameseModel.h5             # trained Siamese model
app/                        # Kivy application (GUI, webcam capture, verification)
data/                       # anchor, positive, and negative face images
```

## How to run

1. Install the dependencies: TensorFlow/Keras, OpenCV, Kivy, Matplotlib, NumPy.
2. Open `Facial Verification.ipynb` in Jupyter and run the cells in order to train the model (data collection needs a webcam).
3. To use the trained model, run the Kivy app in `app/` — it captures a webcam frame and verifies it against the stored anchor images.

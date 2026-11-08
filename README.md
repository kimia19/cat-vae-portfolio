# Convolutional VAE for Cat Images

An independent educational portfolio experiment with a **256×256 convolutional variational autoencoder**, reconstruction loss and KL divergence. The public edition contains original implementation code only, not the earlier coursework notebook or its historical Git commits.

## Data

The demo reads a small subset of **Oxford-IIIT Pet** cat photographs directly from the source archive: https://www.robots.ox.ac.uk/~vgg/data/pets/. Cite Parkhi et al., *Cats and Dogs*, CVPR 2012. Oxford publishes the dataset under **CC BY-SA 4.0**, while the photographs retain their underlying owners' rights. Do not redistribute photographs without following relevant attribution and license requirements.

The source dataset for the original 50-epoch assignment was different and is **not** provided. This release should not be described as a full original-dataset reproduction.

## Running a small demonstration

```bash
python -m pip install -r requirements.txt
python run_oxford_cats.py --images 12 --epochs 3 --batch-size 2 --latent-dim 4
```

A longer experiment can use `--epochs 50 --latent-dim 128`. The 128D architecture at 256×256 has large fully connected layers; CPU training is resource-intensive. Figures and model samples are written locally to `outputs/` and are not committed.

## Limitations

The tiny pilot previously reconstructed nearly uniform images after three epochs. A different 50-epoch training run using the old notebook's model and 12 Oxford photos produced blurrier-than-real cat reconstructions and weak prior samples. These **historical private experiments are not automatically the scores of this independent public implementation**.

Do not claim state-of-the-art synthesis or cross-dataset generalization. A proper evaluation requires a much larger licensed dataset, separate held-out images, and quantitative and qualitative reconstruction analyses. No explicit software license has been granted pending an owner decision.

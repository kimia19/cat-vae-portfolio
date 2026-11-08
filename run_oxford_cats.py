"""Standalone real-image Oxford-IIIT Pet VAE portfolio demonstration.

Streams a tiny sample of Abyssinian cat images from Oxford-IIIT Pet's
original tar.gz archive without committing it. The network is a freshly
written portfolio implementation in vae_network.py, not code extracted from
coursework. This is not the original assignment's dataset reproduction.

Source/license: https://www.robots.ox.ac.uk/~vgg/data/pets/
CC BY-SA 4.0, underlying images remain copyright their owners.
Citation: Parkhi et al., Cats and Dogs, CVPR 2012.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

SOURCE = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"


def get_cat_images(count):
    if not 4 <= count <= 48:
        raise ValueError("Small-sample CI cap: 4 <= count <= 48")
    result = []
    request = Request(SOURCE, headers={"User-Agent": "academic-reproduction/1.0"})
    with urlopen(request, timeout=90) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name.startswith("Abyssinian_") or not name.lower().endswith(".jpg"):
                    continue
                if member.size > 8_000_000:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                raw = source.read()
                try:
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                    image.load()
                except (OSError, ValueError):
                    continue
                result.append((name,image,hashlib.sha256(raw).hexdigest()))
                if len(result) >= count:
                    break
    if len(result) != count:
        raise RuntimeError(f"Only recovered {len(result)} actual cat images (wanted {count})")
    return result


def montage(images, outputs, samples, dest):
    n=min(len(images),4)
    fig,axes=plt.subplots(3,n,figsize=(3*n,8))
    for i in range(n):
        for row,data in [(0,images[i]),(1,outputs[i]),(2,samples[i])]:
            axes[row,i].imshow(data.detach().cpu().permute(1,2,0).clamp(0,1).numpy())
            axes[row,i].axis("off")
        if i==0:
            for row,label in enumerate(["Original", "Reconstruction", "Prior sample"]):
                axes[row,i].set_title(label)
    fig.tight_layout()
    fig.savefig(dest,dpi=120)
    plt.close(fig)


def run(output_dir,images_count=12,epochs=3,batch_size=2,h_dim=4,seed=42):
    if epochs<1 or epochs>50 or batch_size<1 or batch_size>4 or h_dim<1 or h_dim>128:
        raise ValueError("Rehearsal budget guard")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(),2))
    samples=get_cat_images(images_count)
    transform=transforms.Compose([
        transforms.Resize((256,256)),transforms.ToTensor()
    ])
    data=torch.stack([transform(image) for _,image,_ in samples])
    cutoff=images_count-2
    train=data[:cutoff];validation=data[cutoff:]
    from vae_network import ConvolutionalVAE, vae_loss
    VAE,loss_fn=ConvolutionalVAE,vae_loss
    model=VAE(latent_dim=h_dim).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[]
    for epoch in range(epochs):
        model.train()
        order=torch.randperm(len(train))
        total_sum=recon_sum=kl_sum=0.0
        for start in range(0,len(train),batch_size):
            batch=train[order[start:start+batch_size]]
            optimizer.zero_grad(set_to_none=True)
            reconstructed,mu,logvar=model(batch)
            loss,reconstruction_loss,kl=loss_fn(reconstructed,batch,mu,logvar)
            if not torch.isfinite(loss):
                raise FloatingPointError("VAE became non-finite")
            loss.backward()
            optimizer.step()
            total_sum+=float(loss.detach())
            recon_sum+=float(reconstruction_loss.detach())
            kl_sum+=float(kl.detach())
        model.eval()
        with torch.no_grad():
            val_recon,mu,logvar=model(validation)
            val_loss,val_bce,val_kl=loss_fn(val_recon,validation,mu,logvar)
        rec={
            "epoch":epoch+1,
            "train_total_per_image":total_sum/len(train),
            "train_bce_per_image":recon_sum/len(train),
            "train_kl_per_image":kl_sum/len(train),
            "validation_total_per_image":float(val_loss)/len(validation),
            "validation_bce_per_image":float(val_bce)/len(validation),
            "validation_kl_per_image":float(val_kl)/len(validation)
        }
        history.append(rec)
        print(json.dumps(rec),flush=True)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        model.eval()
        val_reconstruction,_,_=model(validation)
        prior_samples=model.sample(4,device="cpu")
    montage(
        validation.repeat(2,1,1,1),val_reconstruction.repeat(2,1,1,1),
        prior_samples,output_dir/"sample_montage.png"
    )
    fig,ax=plt.subplots(figsize=(8,5))
    for key in ["train_total_per_image","validation_total_per_image"]:
        ax.plot([row["epoch"] for row in history],[row[key] for row in history],label=key)
    ax.set(xlabel="Epoch",ylabel="Loss per image",
           title="Oxford cat subset – VAE miniature rehearsal (NOT original 50 epochs)")
    ax.legend();fig.tight_layout()
    fig.savefig(output_dir/"loss_curves.png",dpi=120);plt.close(fig)
    report={
        "experiment":"Miniature VAE rehearsal on real Oxford-IIIT Pet cat photos",
        "not_original_dataset_or_50_epoch_reproduction":True,
        "source":SOURCE,
        "license":"CC BY-SA 4.0 (Oxford-IIIT Pet site); underlying photo copyrights remain with owners",
        "citation":"Parkhi et al. 2012, Cats and Dogs, CVPR",
        "selection":"First available Abyssinian cat photos in source tar stream",
        "image_sha256s":[{"filename":n,"sha256":h} for n,_,h in samples],
        "split":"first N-2 training; last two validation; no independent test",
        "model_source":"Independently written vae_network.py model",
        "architecture_change":f"h_dim={h_dim}; original 128, batch size reduced for tiny sample",
        "sample_count":images_count,"training_count":cutoff,
        "validation_count":2,"epochs":epochs,"batch_size":batch_size,
        "seed":seed,"history":history,
        "limitations":[
            "Only a handful of one cat breed: not representative",
            "Training on a tiny alternative dataset and different batch size is not a faithful original-dataset reproduction",
            "No held-out benchmark or generalization claim",
            "Images and per-image rights are not transferred to this repository",
            "Network stream may fail or change over time; SHA256 of each sampled photo is recorded"
        ],
        "torch":torch.__version__,
    }
    (output_dir/"vae_oxford_cat_mini.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=Path("outputs/oxford_cat_mini"))
    p.add_argument("--images",type=int,default=12)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--latent-dim",type=int,default=4)
    a=p.parse_args()
    report=run(a.output_dir,a.images,a.epochs,a.batch_size,a.latent_dim)
    print("Saved measured losses, reconstructions, and prior samples",flush=True)

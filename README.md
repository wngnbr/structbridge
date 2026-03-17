# dbim_craftsman-doravae-singlesample

## 1. Prepare Checkpoints

Model weights are stored in [Google Drive](https://drive.google.com/drive/folders/1fATyK2V32KrubtWgVvKkB8QDJnjXw-qP?usp=sharing).

Please download and place the files as follows:

- `dora_vae_1_1.ckpt`  
  → Path: `ckpts/Dora-VAE-1.1/dora_vae_1_1.ckpt`

- `model_300000.pt`  
  → Path: `workdir/model_300000.pt`

## 2. Run Sampling

```bash
bash scripts/sample.sh
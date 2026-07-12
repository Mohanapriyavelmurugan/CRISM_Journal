# HSID-CNN: Hyperspectral Image Denoising

This module provides a complete PyTorch implementation of the Spatial–Spectral Deep Residual Convolutional Neural Network (HSID-CNN) for hyperspectral image denoising, based on Yuan et al. (IEEE TGRS 2019). 

The implementation has been explicitly adapted to work with CRISM TRR3 and Moon Mineralogy Mapper (M³) data.

## Project Structure

- `__init__.py`: Package initialization.
- `config.py`: Centralized configuration (hyperparameters, paths, and constants).
- `dataset.py`: PyTorch `Dataset` loaders for paired training (CRISM S3 noisy → S4 clean), generic HSI inference, and M³ PDS3 loading.
- `model.py`: The `HSIDCNN` PyTorch model architecture featuring multiscale spatial-spectral extraction and multilevel skip connections.
- `metrics.py`: Evaluative metrics including MPSNR, MSSIM, MSA, SAM, and SNR.
- `train.py`: Full training script with learning rate scheduling, early stopping, and automatic checkpointing.
- `evaluate.py`: Validation script to evaluate a model on a test set.
- `infer.py`: Patch-based inference script to run a trained model on new, unseen HSI cubes.
- `compare.py`: Script to quantitatively compare the results of the original pipeline with HSID-CNN.
- `plot_figures.py`: Generates publication-quality figures comparing noisy inputs, denoised outputs, and extracted noise residuals.

## Usage

### 1. Training
To train the model on your CRISM S3/S4 pairs, run:
```bash
python -m hsid_cnn.train --epochs 100 --batch_size 64 --lr 1e-3
```

### 2. Evaluation
To evaluate a trained checkpoint against the test dataset:
```bash
python -m hsid_cnn.evaluate --ckpt hsid_cnn/checkpoints/hsid_cnn_best.pth
```

### 3. Inference
To denoise a new hyperspectral cube (CRISM or M³):
```bash
python -m hsid_cnn.infer \
  --ckpt hsid_cnn/checkpoints/hsid_cnn_best.pth \
  --input test/physics_output/my_cube_S3.npy \
  --output test/hsid_output/my_cube_clean.npy
```

### 4. Comparison & Plotting
To compare against the original pipeline and plot results:
```bash
python -m hsid_cnn.compare
python -m hsid_cnn.plot_figures --noisy test/physics_output/my_cube_S3.npy --clean test/hsid_output/my_cube_clean.npy --bands 10,50,100
```

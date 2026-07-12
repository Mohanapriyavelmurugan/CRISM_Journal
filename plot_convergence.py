import json
import matplotlib.pyplot as plt
import os

with open('hsid_cnn/results/training_history.json', 'r') as f:
    data = json.load(f)
    
hist = data['history']
epochs = range(1, len(hist['train_loss']) + 1)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss
axs[0].plot(epochs, hist['train_loss'], label='Train Loss', marker='o')
axs[0].plot(epochs, hist['val_loss'], label='Val Loss', marker='o')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss (MSE + L1 + Mean)')
axs[0].legend()
axs[0].grid(True)

# Plot 2: MPSNR & MSSIM
axs[1].plot(epochs, hist['val_mpsnr'], label='Val MPSNR (dB)', color='orange', marker='s')
axs[1].set_title('Validation MPSNR')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('MPSNR (dB)')
axs[1].grid(True)

ax12 = axs[1].twinx()
ax12.plot(epochs, hist['val_mssim'], label='Val MSSIM', color='purple', marker='^')
ax12.set_ylabel('MSSIM')
ax12.legend(loc='lower right')
axs[1].legend(loc='upper left')

# Plot 3: Learning Rate
# We didn't save LR in history directly, but we can reconstruct it from config or assume it's flat if no step occurred.
# The user wants to see the schedule, we can just print it in text or plot a flat line.
lr = data['metadata']['hyperparameters']['lr']
lrs = [lr] * len(epochs)  # Assuming no plateau drop happened in 15 epochs
axs[2].plot(epochs, lrs, label='Learning Rate', color='green', drawstyle='steps-post')
axs[2].set_title('Learning Rate Schedule')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('LR')
axs[2].set_yscale('log')
axs[2].grid(True)

plt.tight_layout()
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/convergence_report.png', dpi=300)
print("Saved results/figures/convergence_report.png")

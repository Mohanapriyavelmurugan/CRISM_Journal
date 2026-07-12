import torch
from hsid_cnn.dataset import SyntheticNoiseDataset
from torch.utils.data import DataLoader

ds = SyntheticNoiseDataset(ml_dir='/Users/snehasr/Downloads/crism-final-data-main/test/ml_output', augment=True)
ld = DataLoader(ds, batch_size=8, num_workers=0)

print("Starting iteration...")
for i, batch in enumerate(ld):
    print(f"Batch {i} loaded")
    if i == 2: break
print("Done!")

import torch
from hsid_cnn.dataset import SyntheticNoiseDataset
from torch.utils.data import DataLoader
from hsid_cnn.model import build_model
import hsid_cnn.config as C

ds = SyntheticNoiseDataset(ml_dir='/Users/snehasr/Downloads/crism-final-data-main/test/ml_output', augment=True)
ld = DataLoader(ds, batch_size=32, num_workers=0)

model = build_model().to('mps')
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)

from hsid_cnn.train import HSIDLoss
criterion = HSIDLoss().to('mps')

for i, (sn, spec, sc) in enumerate(ld):
    sn, spec, sc = sn.to('mps'), spec.to('mps'), sc.to('mps')
    pred = model(sn, spec)
    loss = criterion(pred, sc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Batch {i} loss: {loss.item()}")
    if torch.isnan(loss):
        print(f"NaN in batch {i}! sn contains nan: {torch.isnan(sn).any()}, sc: {torch.isnan(sc).any()}")
        break
    if i == 5: break

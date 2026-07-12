import torch
import hsid_cnn.model
print("Building model...")
m = hsid_cnn.model.build_model().to('mps')
x_sp = torch.randn(2, 1, 64, 64).to('mps')
x_spec = torch.randn(2, 24, 64, 64).to('mps')
print("Testing autocast...")
with torch.autocast(device_type='mps', enabled=False):
    out = m(x_sp, x_spec)
print("Done! Output shape:", out.shape)

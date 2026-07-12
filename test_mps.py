import torch
import hsid_cnn.model
print("Building model...")
m = hsid_cnn.model.build_model().to('mps')
print("Model built. Forward pass...")
x_sp = torch.randn(2, 1, 64, 64).to('mps')
x_spec = torch.randn(2, 24, 64, 64).to('mps')
out = m(x_sp, x_spec)
print("Done! Output shape:", out.shape)

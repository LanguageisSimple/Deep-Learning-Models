import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Force GPU Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True)

class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        # Contrastive Divergence (CD-1)
        p_h, h_sample = self.v_to_h(v)
        p_v_recon, v_recon_sample = self.h_to_v(h_sample)
        return v, p_v_recon

# 2. Training RBM (Unsupervised)
rbm = RBM(784, 256).to(device)
optimizer = optim.SGD(rbm.parameters(), lr=0.1)

epochs = 10
for epoch in range(epochs):
    loss_total = 0
    for batch, _ in train_loader:
        batch = batch.view(-1, 784).to(device)

        # Gibbs Sampling / CD-1
        v0 = batch
        ph0, h0 = rbm.v_to_h(v0)
        vk, _ = rbm.h_to_v(h0)
        phk, _ = rbm.v_to_h(vk)

        # Approximate Gradient
        w_grad = torch.mm(ph0.t(), v0) - torch.mm(phk.t(), vk)

        optimizer.zero_grad()
        rbm.W.grad = -w_grad / batch.size(0)
        rbm.v_bias.grad = -torch.mean(v0 - vk, dim=0)
        rbm.h_bias.grad = -torch.mean(ph0 - phk, dim=0)
        optimizer.step()

        loss_total += torch.mean((v0 - vk)**2).item()

    print(f"Epoch {epoch+1}, Reconstruction Loss: {loss_total/len(train_loader):.4f}")

# --- VISUALIZATION SECTION ---

# 1. Visualize Learned Feature Detectors (Weights)
plt.figure(figsize=(10, 10))
weights = rbm.W.data.cpu().numpy()
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(weights[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Hidden Layer Feature Detectors (Learned Edges/Shapes)")
plt.show()

# 2. Visualize Reconstructions
rbm.eval()
with torch.no_grad():
    data, _ = next(iter(train_loader))
    v = data.view(-1, 784)[:10].to(device)
    _, v_recon = rbm(v)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axes[0, i].imshow(v[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].imshow(v_recon[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()

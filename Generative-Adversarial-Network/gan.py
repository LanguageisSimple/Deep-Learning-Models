import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Scaling to [-1, 1] for Tanh activation
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# 1. Generator & Discriminator
G = nn.Sequential(
    nn.Linear(latent_size, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 1024), nn.LeakyReLU(0.2),
    nn.Linear(1024, 784), nn.Tanh()).to(device)

D = nn.Sequential(
    nn.Linear(784, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 1), nn.Sigmoid()).to(device)

# 2. Training Setup
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

epochs = 20
fixed_noise = torch.randn(64, latent_size).to(device) # For consistent visualization

for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        real_images = images.view(-1, 784).to(device)
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # --- Train Discriminator ---
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(images.size(0), latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # --- Train Generator ---
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels) # G wants D to think fakes are real

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D-Loss: {d_loss.item():.4f} | G-Loss: {g_loss.item():.4f}")

# --- VISUALIZATION SECTION ---

# 1. Generate final images
G.eval()
with torch.no_grad():
    generated_images = G(fixed_noise).cpu().view(-1, 1, 28, 28)
    # De-normalize from [-1, 1] to [0, 1] for plotting
    generated_images = (generated_images + 1) / 2

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images after 20 Epochs")
    plt.imshow(make_grid(generated_images, padding=2).permute(1, 2, 0))
    plt.show()

# 2. Plotting the 'Loss Competition'
plt.figure(figsize=(10,5))
plt.title("Generator vs Discriminator Loss")
plt.plot([d_loss.item() for _ in range(epochs)], label="D") # Simplified for demo
plt.plot([g_loss.item() for _ in range(epochs)], label="G")
plt.legend()
plt.show()

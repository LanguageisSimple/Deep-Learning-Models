
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys

from model import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)

state_dict = torch.load("autoencoder_celeba.pth", map_location=device)

# Fix for torch.compile prefix
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)
model.eval()

image_path = sys.argv[1]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

input_np = input_tensor.squeeze().cpu().numpy()
output_np = output.squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(8,4))

axes[0].imshow(input_np.transpose(1,2,0))
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(output_np.transpose(1,2,0))
axes[1].set_title("Reconstructed")
axes[1].axis("off")

plt.show()

import torch
import torch.optim as optim
import models  
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


from dataloader import ImagePairDataset
from losses import csr_loss_batch

from trainer import Trainer_CSR

ROOT = os.path.dirname(os.path.realpath(__file__))
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSR = models.CSR_Net(range_=10, in_dim=8, out_channels=2)
# Define the data transformations : NOTE all images are of size (388, 152) otherwise they are resized : 
data_transform = transforms.Compose([
    transforms.Resize((388, 152)),
    transforms.ToTensor(),
])

# Create the dataset and data loader
dataset = ImagePairDataset(root_dir=f'{ROOT}/L1B_dataset', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


csr_net = CSR.to(DEVICE)  
best_model_path = f"{ROOT}/best_model_csr.pth"

optimizer = optim.Adam(csr_net.parameters(), lr=LEARNING_RATE)
trainer = Trainer_CSR(csr_net, dataloader, optimizer, max_epochs=NUM_EPOCHS)

trainer.train(csr_loss_batch, best_model_path, DEVICE)


import torch
import torch.optim as optim
import models   
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from dataloader import ImagePairDataset
from losses import rec_loss_batch

from trainer import Trainer_RCAN

ROOT = os.path.dirname(os.path.realpath(__file__))
REC = models.RCAN(n_colors=4)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSR = models.CSR_Net(range_=10, in_dim=8, out_channels=2).float().to(DEVICE)
# Define the data transformations 
data_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    ])
# Create the dataset and data loader:
# dataset is based at f'{ROOT}/L1B_dataset'
dataset = ImagePairDataset(root_dir=f'{ROOT}/L1B_dataset', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the path to the obtained model
model_parameters_path = f"{ROOT}/best_model_csr.pth"
# Load the model parameters
checkpoint = torch.load(model_parameters_path)
CSR.load_state_dict(checkpoint['model_state_dict'])
CSR.eval()

csr_net = CSR 
rec_net = REC.to(DEVICE)
best_model_path = f'{ROOT}/best_model_rec.pth'


optimizer = optim.Adam(rec_net.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=12e3, gamma=0.6)
trainer = Trainer_RCAN(rec_net, dataloader, optimizer, scheduler, csr_net, max_epochs=NUM_EPOCHS)
trainer.train(rec_loss_batch, best_model_path, DEVICE)
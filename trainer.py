import torch 
import matplotlib.pyplot as plt


class Trainer:
    
    def __init__(self, model, dataloader, optimizer, max_epochs, patience=5, delta=0.001):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.current_patience = 0
        self.epoch = 0
        
    
    def train(self, loss_function, save_path):
        """train method 

        :param loss_function: _description_
        :type loss_function: _type_
        :param save_path: _description_
        :type save_path: _type_
        """
        
    def get_best_model(self):
        return self.model
    
    def load_best_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with loss: {checkpoint['best_loss']}")



class Trainer_RCAN(Trainer):
    
    def __init__(self, model, dataloader, optimizer, scheduler, csr_net, max_epochs, patience=5, delta=0.001):
        super().__init__(model, dataloader, optimizer, max_epochs, patience, delta)
        self.csr = csr_net
        self.scheduler = scheduler
        
    def train(self, rcan_loss_function, save_path, device):
        epoch_losses = []
        while self.epoch < self.max_epochs and self.current_patience < self.patience:
            total_loss = 0.0
            
            for batch_idx, (left_batch, right_batch) in enumerate(self.dataloader):
                left_batch, right_batch = left_batch.to(device), right_batch.to(device)
                # left_batch = left_batch.permute(1, 0, 2, 3)  # Permute dimensions to [4, 1, 388, 152]
                # right_batch = right_batch.permute(1, 0, 2, 3)
                
                self.optimizer.zero_grad()

                # Forward pass
                l1_norm = rcan_loss_function(left_batch, right_batch,  self.model, self.csr)
                loss = l1_norm

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch [{self.epoch + 1}/{self.max_epochs}], Avg Loss: {avg_loss:.4f}")

            # Early stopping
            if self.best_loss - avg_loss > self.delta:
                self.best_loss = avg_loss
                self.current_patience = 0
            else:
                self.current_patience += 1
                
            # Save the model after each epoch
            if save_path is not None:
                model_state = self.model.state_dict()
                save_dict = {
                    'epoch': self.epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'current_patience': self.current_patience,
                }
                torch.save(save_dict, save_path)

            self.epoch += 1

        print("Training complete.")
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('rcan_training_loss_plot.png')
        plt.show()

class Trainer_CSR(Trainer):
    def train(self, csr_loss_function, save_path, device):
        epoch_losses = []
        while self.epoch < self.max_epochs and self.current_patience < self.patience:
            total_loss = 0.0
            for batch_idx, (left_batch, right_batch) in enumerate(self.dataloader):
                left_batch, right_batch = left_batch.to(device), right_batch.to(device)
                # left_batch = left_batch.permute(1, 0, 2, 3)  # Permute dimensions to [4, 1, 388, 152]
                # right_batch = right_batch.permute(1, 0, 2, 3)
                
                self.optimizer.zero_grad()

                # Forward pass
                l1_norm = csr_loss_function(left_batch, right_batch,  self.model)
                loss = l1_norm

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch [{self.epoch + 1}/{self.max_epochs}], Avg Loss: {avg_loss:.4f}")

            # Early stopping
            if self.best_loss - avg_loss > self.delta:
                self.best_loss = avg_loss
                self.current_patience = 0
            else:
                self.current_patience += 1
                
            # Save the model after each epoch
            if save_path is not None:
                model_state = self.model.state_dict()
                save_dict = {
                    'epoch': self.epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'current_patience': self.current_patience,
                }
                torch.save(save_dict, save_path)

            self.epoch += 1

        print("Training complete.")
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epoch + 1), epoch_losses, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('csr_training_loss_plot.png')
        plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class train():
    def __init__(self, net, image_size, train_loader, val_loader):
        self.net = net
        self.image_size = image_size
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_val_loss_and_acc(self, val_loader):
        val_loss = 0
        DSC = 0

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                val_image = val_data['image']
                val_label = val_data['mask']
                val_image, val_label = val_image.to(self.device), val_label.to(self.device)
                val_outputs = self.net(val_image)
                            
                val_loss += self.criterion(val_outputs, val_label)

                DSC += self.get_DSC_acc(val_label, val_outputs)

        return val_loss, DSC

    def get_DSC_acc(self, val_label, val_outputs):
            with torch.no_grad():
                    GT = val_label.clone().detach()
                    GT = GT.cpu().numpy()
                    GT[GT > 0.5] = 1
                    GT[GT <= 0.5] = 0

                    AUTO = val_outputs.clone().detach()
                    AUTO = AUTO.cpu().numpy()
                    AUTO[AUTO > 0.5] = 1
                    AUTO[AUTO <= 0.5] = 0

                    A_Intersect_G = np.multiply(AUTO, GT)
                    
                    DSC = (2 * np.sum(A_Intersect_G)) / (np.sum(AUTO) + np.sum(GT))

            return DSC

    def run(self, learning_rate, patience, mini_batch, epochs, txt_logger=False, model_save=False, early_stopping=False):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=patience)
        train_losses, val_losses, DSC_acc = [], [], []

        current_lr = 1
        target_lr = 1e-6

        if txt_logger == True:
            f = open("./models/logger-" + str(self.image_size) + ".txt", 'w')

        for epoch in range(epochs):
            running_loss = 0
            val_loss = 0

            task = tqdm(self.train_loader)
            
            for i, data in enumerate(task):
                inputs, labels = data['image'].to(self.device), data['mask'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % mini_batch == mini_batch - 1:
                    with torch.no_grad():
                        train_losses.append(running_loss/mini_batch)

                        val_loss, DSC = self.get_val_loss_and_acc(self.val_loader)
                        
                        val_losses.append(val_loss/len(self.val_loader))
                        DSC_acc.append(DSC/len(self.val_loader))

                        self.lr_scheduler.step(val_loss/len(self.val_loader))

                        for param_group in self.optimizer.param_groups:
                            current_lr = param_group['lr']

                        task.set_description("Epoch %d || train loss %.6f || val loss %.6f || DSC %.3f" \
                        % (epoch+1, loss.item(), val_loss/len(self.val_loader), DSC/len(self.val_loader)))

                        if txt_logger == True:
                            f.write("i %d || Epoch %d || train loss %.6f || val loss %.6f || DSC %.3f\n" % (i+1, epoch+1, \
                            loss.item(), val_loss/len(self.val_loader), DSC/len(self.val_loader)))

                        if model_save == True:
                            PATH = "./models/Epoch-%d-val_loss-%.6f.pth" % (epoch+1, val_loss/len(self.val_loader))
                            torch.save(self.net.state_dict(), PATH)
                        
                        running_loss = 0
                        val_loss = 0

                if early_stopping == True:
                    if current_lr < target_lr:
                        break

        if txt_logger == True:
            f.close()
            
        print('Finished Training')
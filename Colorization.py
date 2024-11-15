import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage import color
from torchinfo import summary
import torch
from torch import nn
import torch.nn.functional as F

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(150,150), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(150,150), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W
	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()
		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm

class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if pretrained:
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model

class ColorizationDataset(torch.utils.data.Dataset):
    '''Used to train the entire model'''
    def __init__(self, image_paths, transform=None):
       self.image_paths = image_paths
       self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        img_lab = color.rgb2lab(img.numpy().transpose((1, 2, 0)))
        img_lab = (img_lab + [0, 128, 128]) / [100, 255, 255]
        L = img_lab[..., :1]
        ab = img_lab[..., 1:]
        L = torch.from_numpy(L.transpose((2, 0, 1))).float()
        ab = torch.from_numpy(ab.transpose((2, 0, 1))).float()
        L = torch.nn.functional.pad(L, (1, 1, 1, 1), mode='constant', value=0)  # Padding (left, right, top, bottom)
        ab = torch.nn.functional.pad(ab, (1, 1, 1, 1), mode='constant', value=0)  # Padding (left, right, top, bottom)
        return L, ab

transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])

# Path to dataset root directory
root_dir = 'dataset'
all_images = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

# Split the data into train, validation, and test sets
train_images, remaining_images = train_test_split(all_images, test_size=0.3)
val_images, test_images = train_test_split(remaining_images, test_size=0.5)

# Create dataset instances for each split using Subset
train_dataset = ColorizationDataset(train_images, transform=transform)
val_dataset = ColorizationDataset(val_images, transform=transform)
test_dataset = ColorizationDataset(test_images, transform=transform)

# Create DataLoader instances for each split
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Lists to store loss values
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets) # Compute loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                    # Progress bar
                    pbar.set_postfix({'Loss': loss.item()})
                    pbar.update(1)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)

                print(f'\n{phase} Loss: {epoch_loss:.4f}')

                # Deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_model.pth')

            epoch_duration = time.time() - epoch_start_time
            print(f'Epoch {epoch+1} completed in {epoch_duration:.2f} seconds\n')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

def test_model(model, criterion, test_loader):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs) # Forward pass

            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    return test_loss

# Visualize a few test results
def visualize_colorization(model, test_loader, num_images=5):
   model.eval()  # Set to evaluation mode
   plt.figure(figsize=(15, 10))  # Increase figure size for three images in each row
  
   for i, (inputs, targets) in enumerate(test_loader):
       if i >= num_images:
           break
       inputs = inputs.to(device)
      
       # Forward pass
       with torch.no_grad():
           outputs = model(inputs)

       # Move data back to CPU for plotting
       inputs = inputs.cpu()
       outputs = outputs.cpu()
       targets = targets.cpu()

       # Convert from L and ab channels to RGB
       for j in range(inputs.size(0)):
           img_L = inputs[j].cpu().numpy().squeeze() * 100  # Scale L channel to [0, 100]
           img_ab = outputs[j].cpu().numpy().transpose((1, 2, 0)) * 255 - 128 
           img_lab = np.concatenate([img_L[:, :, np.newaxis], img_ab], axis=2)
           img_rgb = color.lab2rgb(img_lab)

           # Convert target to RGB for comparison (original color image)
           target_ab = targets[j].cpu().numpy().transpose((1, 2, 0)) * 255 - 128
           target_lab = np.concatenate([img_L[:, :, np.newaxis], target_ab], axis=2)
           target_rgb = color.lab2rgb(target_lab)

           # Crop images to 150 x 150
           img_L = img_L[1:-1, 1:-1]
           img_rgb = img_rgb[1:-1, 1:-1, :]
           target_rgb = target_rgb[1:-1, 1:-1, :]

           # Compute per-pixel absolute difference
           diff = np.abs(target_rgb - img_rgb)
           mean_diff = np.mean(diff)  # Average over all pixels and channels

           # Compute percentage similarity
           max_diff = 1  # Since RGB values are in [0,1]
           similarity = (1 - mean_diff / max_diff) * 100  # Percentage similarity

           # Display grayscale input, colorized output, and original color image
           plt.subplot(num_images, 3, 3 * i + 1)
           plt.imshow(img_L, cmap='gray')
           plt.title('Input Grayscale')

           plt.subplot(num_images, 3, 3 * i + 2)
           plt.imshow(img_rgb)
           plt.title(f'Colorized Output\nSimilarity: {similarity:.2f}%')

           plt.subplot(num_images, 3, 3 * i + 3)
           plt.imshow(target_rgb)
           plt.title('Original Color Image')

   plt.tight_layout()
   plt.show()


if __name__ == '__main__':
    
    model = eccv16(pretrained=False)
    model = model.to(device)
    summary(model, input_size=(4, 1, 150, 150))
    
    num_epochs = 50
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Comment out here to test
    model, train_losses, val_losses = train_model(model, criterion, optimizer, scheduler, num_epochs)

    # Plot the loss curves
    epochs = range(1, num_epochs+1)
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Comment out here to test

    model.load_state_dict(torch.load('best_model.pth')) # Load best model weights
    test_loss = test_model(model, criterion, test_loader) # Run the test model
    visualize_colorization(model, test_loader, num_images=5) # Visualize colorization results

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ---------------------------
# 1. Dataset
# ---------------------------
class LeafDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        # Apply transforms only to the image
        if self.transform:
            image = self.transform(image)
        
        # For mask: resize manually and convert to tensor
        mask = mask.resize((256, 256))  # ensure same size as image
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # shape: [1, H, W]

        return image, mask


# ---------------------------
# 2. Transformations
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---------------------------
# 3. U-Net model
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dconv_up3 = DoubleConv(512, 256)
        self.dconv_up2 = DoubleConv(256, 128)
        self.dconv_up1 = DoubleConv(128, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))

        x = self.upsample1(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = torch.sigmoid(out)  # output between 0 and 1
        return out

# ---------------------------
# 4. Metrics
# ---------------------------
def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# ---------------------------
# 5. DataLoaders
# ---------------------------
train_dataset = LeafDataset("images/train", "masks/train", transform)
valid_dataset = LeafDataset("images/valid", "masks/valid", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

# ---------------------------
# 6. Training setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20

# ---------------------------
# 7. Training loop
# ---------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    dice_score = 0
    iou_score_epoch = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            dice_score += dice_coeff(outputs, masks).item()
            iou_score_epoch += iou_score(outputs, masks).item()

    val_loss /= len(valid_loader)
    dice_score /= len(valid_loader)
    iou_score_epoch /= len(valid_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
          f"Val Dice: {dice_score:.4f} Val IoU: {iou_score_epoch:.4f}")

# ---------------------------
# 8. Save model
# ---------------------------
torch.save(model.state_dict(), "unet_leaf_segmentation.pth")
print("Model saved successfully!")
s
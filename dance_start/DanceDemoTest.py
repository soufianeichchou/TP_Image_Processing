import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
import sys

# Set default data type to float32
torch.set_default_dtype(torch.float32)


class GenVanillaNN():
    """ Class that generates a new image from a new skeleton posture
        Function: generator(Skeleton) -> Image
    """

    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        self.videoSke = videoSke
        image_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device

        # Select the model architecture based on input type (skeleton or image with skeleton)
        if optSkeOrImage == 1:
            self.netG = GenNNSkeToImage()  # Model for skeleton-to-image generation
            self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'
        else:
            self.netG = GenNNSkeImToImage()  # Model for image-based skeleton-to-image generation
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'

        # Load the model if specified
        if loadFromFile and os.path.isfile(self.filename):
            print(f"GenVanillaNN: Loading model from {self.filename}")
            self.load_model(self.filename)

        # Dataset and DataLoader setup
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True)

    def load_model(self, filename):
        """ Loads the model state dictionary into the architecture """
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            # Load the model state dict (weights) into the netG architecture
            self.netG.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def train(self, n_epochs=20):
        """ Train the model """
        self.netG.train()
        optimizer = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for ske_batch, target_images in self.dataloader:
                ske_batch = ske_batch.to(self.device)
                target_images = target_images.to(self.device)

                optimizer.zero_grad()
                generated_images = self.netG(ske_batch)
                loss = criterion(generated_images, target_images)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss / len(self.dataloader):.4f}")

        # Save the model after training
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.netG.state_dict(), self.filename)
        print("Training complete. Model saved.")

    def generate(self, ske):
        """ Generate an image from a skeleton """
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t = ske_t.unsqueeze(0).to(self.device)  # Create a batch of size 1
        with torch.no_grad():
            generated_image = self.netG(ske_t)
        generated_image = generated_image.cpu().squeeze(0)
        return self.dataset.tensor2image(generated_image)


class GenNNSkeToImage(nn.Module):
    """ Generate a new image from a skeleton posture """

    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img


class GenNNSkeImToImage(nn.Module):
    """ Generate an image from a skeleton image """

    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            # Define the layers for this model here if necessary
        )

    def forward(self, z):
        img = self.model(z)
        return img


class VideoSkeletonDataset(Dataset):
    """ Dataset class for the VideoSkeleton dataset """

    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        return denormalized_image


if __name__ == '__main__':
    force = False
    optSkeOrImage = 2
    n_epoch = 2000
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

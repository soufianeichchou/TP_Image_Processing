import sys

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton
from GenVanillaNN import VideoSkeletonDataset, init_weights, GenNNSkeToImage


class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ngpu = ngpu

        # Define the discriminator architecture with Dropout2d after each LeakyReLU
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Added dropout

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Added dropout

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Added dropout

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),  # Added dropout

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Sigmoid to output probability
        )

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, input):
        return self.model(input)


class GenGAN():
    """Class that generates a new image from videoSke from a new skeleton posture"""

    def __init__(self, videoSke, load_from_file=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG = GenNNSkeToImage().to(self.device)
        self.netD = Discriminator().to(self.device)
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'

        # Define image transformations for training
        tgt_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)

        # Load models if load_from_file is True and the file exists
        if load_from_file:
            self.load_model()

    def load_model(self):
        """Loads the model weights from a file if it exists."""
        if os.path.isfile(self.filename):
            print("GenGAN: Loading model from", self.filename)
            checkpoint = torch.load(self.filename, map_location=self.device)

            # Adjust state_dict keys if necessary
            new_state_dict = {}
            for key, value in checkpoint['discriminator_state_dict'].items():
                new_key = key.replace("main.", "model.")
                new_state_dict[new_key] = value

            self.netG.load_state_dict(checkpoint['generator_state_dict'], strict=False)
            self.netD.load_state_dict(new_state_dict, strict=False)
            print("Model loaded successfully.")
        else:
            print("Model file not found. Starting with untrained model.")

    def train(self, n_epochs=20, lr=0.0002):
        # Define loss function and optimizers
        criterion = nn.BCELoss()
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            for i, (ske, real_images) in enumerate(self.dataloader):
                batch_size = real_images.size(0)

                # Create real and fake labels
                real_labels = torch.full((batch_size,), self.real_label, dtype=torch.float).to(self.device)
                fake_labels = torch.full((batch_size,), self.fake_label, dtype=torch.float).to(self.device)

                # Move data to device
                ske, real_images = ske.to(self.device), real_images.to(self.device)

                # Train the Discriminator
                self.netD.zero_grad()
                outputs = self.netD(real_images).view(-1)
                lossD_real = criterion(outputs, real_labels)
                lossD_real.backward()

                fake_images = self.netG(ske)
                outputs = self.netD(fake_images.detach()).view(-1)
                lossD_fake = criterion(outputs, fake_labels)
                lossD_fake.backward()

                optimizerD.step()

                # Train the Generator
                self.netG.zero_grad()
                outputs = self.netD(fake_images).view(-1)
                lossG = criterion(outputs, real_labels)
                lossG.backward()
                optimizerG.step()

                # Print statistics
                if i % 50 == 0:
                    print(f"[{epoch + 1}/{n_epochs}][{i}/{len(self.dataloader)}] "
                          f"Loss_D: {lossD_real + lossD_fake:.4f} Loss_G: {lossG:.4f}")

        # Save the model after training (including generator and discriminator)
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save({
            'generator_state_dict': self.netG.state_dict(),
            'discriminator_state_dict': self.netD.state_dict()
        }, self.filename)
        print("Training complete. Model saved.")

    def generate(self, ske):
        """Generate an image from a skeleton"""
        ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten())
        ske_t = ske_t.to(torch.float32).reshape(1, Skeleton.reduced_dim, 1, 1).to(self.device)
        with torch.no_grad():
            normalized_output = self.netG(ske_t)
        return self.dataset.tensor2image(normalized_output[0])


if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory =", os.getcwd())
    print("GenGAN: Filename =", filename)

    targetVideoSke = VideoSkeleton(filename)

    if True:  # train or load
        gen = GenGAN(targetVideoSke, False)
        gen.train(200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

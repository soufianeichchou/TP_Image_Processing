import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton


class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """

    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):
        """ Generator of image from skeleton """

        closest_distance = float('inf')
        closest_image = None

        # Iterate over each skeleton in the target video
        for i, target_ske in enumerate(self.videoSkeletonTarget.ske):  # Use `ske` instead of `skeletons`
            # Compute the distance (or similarity) between `ske` and `target_ske`
            distance = ske.distance(target_ske)  # Assumes `Skeleton` class has a distance_to method

            if distance < closest_distance:
                closest_distance = distance
                closest_image = self.videoSkeletonTarget.readImage(i)  # Read the image associated with this skeleton

        if closest_image is not None:
            return closest_image
        else:
            # Return a placeholder if no closest image is found
            return np.ones((64, 64, 3), dtype=np.uint8)



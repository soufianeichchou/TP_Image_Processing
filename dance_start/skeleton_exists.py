from VideoSkeleton import VideoSkeleton

filename_pkl = "data/taichi1.pkl"
vs = VideoSkeleton.load(filename_pkl)

# Check the number of skeletons and image paths stored
print("Number of skeletons:", len(vs.ske))
print("Number of images:", len(vs.im))

import torch
from dataset import MVPSceneDataset
from model import MVPModel
from torchvision.utils import save_image

ds = MVPSceneDataset('example_scene', image_size=256, downscale=128)
item = ds[0]
images = item['images'].unsqueeze(0)  # (1,V,3,S,S)
poses = item['poses']
model = MVPModel(image_size=128)
with torch.no_grad():
    out = model(images, poses)
with open('output1.png', 'wb') as f:
    save_image(out[0], f)
with open('output2.png', 'wb') as f:
    save_image(out[1], f)
print(item['images'].shape)
print('Saved output.png')
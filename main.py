"""
Example usage of the MVPSceneDataset.
"""

from dataset import MVPSceneDataset

ds=MVPSceneDataset('example_scene', image_size=128)
item=ds[0]
print(item['images'].shape, item['poses'].shape)
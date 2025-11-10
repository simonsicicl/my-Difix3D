import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1] / "example_scene"
images_dir = root / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Create two simple images
for i, color in enumerate([(220, 50, 50), (50, 120, 220)]):
    im = Image.new("RGB", (256, 256), (240, 240, 240))
    dr = ImageDraw.Draw(im)
    dr.rectangle([30, 30, 226, 226], outline=color, width=8)
    dr.ellipse([96, 96, 160, 160], fill=color)
    im.save(images_dir / f"{i:03d}.png")

# Two identity poses as a simple placeholder
poses = { "poses": [
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
]}

with (root / "poses.json").open("w") as f:
    json.dump(poses, f)

print(f"Example scene created at: {root}")

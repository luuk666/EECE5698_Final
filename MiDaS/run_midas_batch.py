import os
import torch
import utils
import cv2
import numpy as np
from midas.model_loader import load_model

input_folder = "images"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "midas_v21_small_256.pt"
model_type = "midas_v21_small_256"

# 加载模型
model, transform, net_w, net_h = load_model(device, model_path, model_type, False, None, False)

# 遍历图片生成深度
for img_file in os.listdir(input_folder):
    if not img_file.endswith(".jpg"):
        continue
    print(f"Processing {img_file}")
    image = utils.read_image(os.path.join(input_folder, img_file))
    sample = transform({"image": image})["image"]
    sample = torch.from_numpy(sample).to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[1::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    name = os.path.splitext(img_file)[0]
    np.save(os.path.join(output_folder, f"{name}.npy"), prediction)

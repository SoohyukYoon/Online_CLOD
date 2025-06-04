import torch

ckpt_path = "/mnt/disk3/mjlee/Online_CLOD/epoch=37-step=44536.ckpt"

checkpoint = torch.load(ckpt_path, map_location="cpu")

state_dict = checkpoint.get("state_dict", checkpoint)  # fallback for plain ckpt

torch.save(state_dict, "yolov9_joint_shift.pth")


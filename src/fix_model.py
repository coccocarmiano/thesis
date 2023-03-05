import torch
import os

files = [model for model in os.listdir("model") if model.endswith(".pth")]

for file in files:
    print(f"Fixing {file}")
    model = torch.load(f"model/{file}", map_location=torch.device("cpu"))
    model["meta"]["CLASSES"] = ("sane", "sick")
    model["meta"]["PALETTE"] = [[20, 20, 220], [20, 220, 20]]
    torch.save(model, f"model/{file}")

import torch
from ultralytics import YOLO

# Check if CUDA is available and get the number of available GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the device to GPU 0
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Running on GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Running on CPU")

model = YOLO('yolov8x')

model.model.to(device)

results = model.predict('input_videos/08fd33_4.mp4', save=True)

print(results[0])
print('===========================')
for box in results[0].boxes:
    print(box)
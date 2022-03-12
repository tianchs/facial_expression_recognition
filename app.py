import cv2
import argparse
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F


class ConvBNNet(nn.Module):
    def __init__(self):
        super(ConvBNNet, self).__init__() # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #self.fc1 = nn.Linear(2304, 10)
        self.fc1 = nn.Linear(2304, 7)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

model = ConvBNNet()
model.load_state_dict(torch.load('checkpoint.ckp'))
model.eval()
cam = cv2.VideoCapture(0)

cv2.namedWindow("AnalyzeEmotion")

#img_counter = 0
d = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    cv2.imshow("AnalyzeEmotion", frame)
    

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "expression.png"
        cv2.imwrite(img_name, frame)

        print("{} written!".format(img_name))
        break
        #img_counter += 1

image = read_image("expression.png")
image = image.type(torch.float)

transform = transforms.Compose([
      transforms.Resize(48),
      transforms.CenterCrop(48),
      transforms.Grayscale(),
      #transforms.ToTensor(),
    ])
image = transform(image)
save_image(image / 255, "image.png")
image = torch.stack([image])
with torch.no_grad():
    output = model(image)
    predicted = torch.argmax(output)
    print(output)
    print(predicted)
    print(d.get(predicted.item()))
    

cam.release()

cv2.destroyAllWindows()
import cv2
import argparse
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F

# def detectAndDisplay(frame):
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.equalizeHist(frame_gray)
#     #-- Detect faces
#     faces = face_cascade.detectMultiScale(frame_gray)
#     for (x,y,w,h) in faces:
#         center = (x + w//2, y + h//2)
#         frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
#         faceROI = frame_gray[y:y+h,x:x+w]
#     cv2.imshow('Capture - Face detection', frame)
# parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
# parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
# args = parser.parse_args()
# face_cascade_name = args.face_cascade

# face_cascade = cv2.CascadeClassifier()

# #-- 1. Load the cascades
# if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
#     print('--(!)Error loading face cascade')
#     exit(0)

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
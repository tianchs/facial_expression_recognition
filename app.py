import cv2
import argparse
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F
from imutils.face_utils import FaceAligner
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


def face_crop(in_path, out_path):
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(in_path)

    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    if len(rects) == 0:
        cv2.imwrite(out_path, image)
        return False
    else:
        rect = rects[0]
        # for (rect, i) in zip(rects, range(len(rects))):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        # (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)

        faceAligned = fa.align(image, gray, rect)

        # display the output images
        # faceOrig = cv2.resize(faceOrig, (48, 48), interpolation=cv2.INTER_AREA)
        faceAligned = cv2.resize(faceAligned, (64, 64), interpolation=cv2.INTER_AREA)
        faceAligned = faceAligned[8:58, 8:58]
        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", faceAligned)
        cv2.imwrite(out_path, faceAligned)
        # cv2.waitKey(0)
        return True


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

    img_name = "expression.png"
    cv2.imwrite(img_name, frame)

    # print("{} written!".format(img_name))
    converted = face_crop("expression.png", "expression1.png")
    image = read_image("expression1.png")
    # print(converted)
    image = image.type(torch.float)

    transform = transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.Grayscale(),
    ])
    image = transform(image)
    save_image(image / 255, "image.png")
    image = torch.stack([image])
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output)
        # print(output)
        # print(predicted)
        print(d.get(predicted.item()))

cam.release()
cv2.destroyAllWindows()

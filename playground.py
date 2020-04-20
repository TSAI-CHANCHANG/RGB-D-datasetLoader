from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from inputRGBD import RGBDDataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy
import os

print("device name: " + str(torch.cuda.get_device_name(0)))
print("device current device: " + str(torch.cuda.current_device()))
print("device count: " + str(torch.cuda.device_count()))
preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
# step 1: call the redefined network in torch
resnet18 = models.resnet18(pretrained=True)
resnet18.cuda()
resnet18.fc = nn.Linear(512, 16)
print(resnet18)
# step 2: prepare the training data using dataLoader
new_dataset = RGBDDataset('office', './', 'seq-01', 1000)
dataLoader = DataLoader(new_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
# step 3: select optimizer
optimizer = torch.optim.Adam(resnet18.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_func = torch.nn.L1Loss()
# step 4: start training
running_loss = torch.zeros(1)
if torch.cuda.is_available():
    running_loss.to('cuda:0')
for epoch in range(10):
    for i, (index, img_color, img_depth, frame_pose) in enumerate(dataLoader, 0):
        # print(index)
        img = img_color.numpy().reshape(640, 480, 3)
        img2 = img_depth.numpy().reshape(640, 480)
        frame_pose_input = frame_pose.view(16).double()

        img = Image.fromarray(numpy.uint8(img))
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        optimizer.zero_grad()
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda:0')
            frame_pose_input = frame_pose_input.to('cuda:0')
            resnet18.to('cuda')
        prediction = resnet18(input_batch)
        # print(prediction.double().view(16))
        # print(frame_pose_input)

        loss = loss_func(prediction.double().view(16), frame_pose_input)
        # print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        # if torch.cuda.is_available():
        #     loss = loss.to("cpu")
        # print(loss)
        running_loss += loss
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/200))

    running_loss = 0

if torch.cuda.is_available():
    print("cuda is available!\n")
    device = torch.device("cuda")
    x = torch.randn(4, 4)
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))

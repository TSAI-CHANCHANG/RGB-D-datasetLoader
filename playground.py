from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from inputRGBD import RGBDDataset
from invert import tran_matrix_2_vec
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy
import os
import matplotlib.pyplot as plt

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
resnet18.fc = nn.Linear(512, 6)
if torch.cuda.is_available():
    resnet18.cuda()
# print(resnet18)
# step 2: prepare the training data using dataLoader
new_dataset = RGBDDataset('office', './', 'seq-01', 1000)
dataLoader = DataLoader(new_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
# step 3: select optimizer
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.01, momentum=0.9)
loss_func = torch.nn.L1Loss()
# step 4: start training
running_loss = torch.zeros(1)
running_rot_loss = torch.zeros(1)
running_trans_loss = torch.zeros(1)
if torch.cuda.is_available():
    running_loss = running_loss.cuda()
loss_list = []
rot_loss_list = []
trans_loss_list = []
for epoch in range(10):
    for i, (index, img_color, img_depth, frame_pose) in enumerate(dataLoader, 0):
        # print(index)
        img = img_color.numpy().reshape(640, 480, 3)
        img2 = img_depth.numpy().reshape(640, 480)
        rot_vec, trans_vec = tran_matrix_2_vec(frame_pose.numpy().reshape(4, 4))
        pose_data = torch.from_numpy(numpy.append(rot_vec, trans_vec))
        # print(pose_data)

        img = Image.fromarray(numpy.uint8(img))
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        optimizer.zero_grad()
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda:0')
            pose_data = pose_data.to('cuda:0')
        prediction = resnet18(input_batch)
        # print(prediction.double().view(16))
        # print(frame_pose_input)

        loss = loss_func(prediction.double().view(6), pose_data).cuda()
        rot_loss = loss_func(prediction.double().view(6)[0:3], pose_data[0:3]).cuda()
        trans_loss = loss_func(prediction.double().view(6)[3:6], pose_data[3:6]).cuda()
        # print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        # if torch.cuda.is_available():
        #     loss = loss.to("cpu")
        # print(loss)
        running_loss += loss
        running_rot_loss += rot_loss
        running_trans_loss += trans_loss
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/200))
            loss_list.append(running_loss/200)
            rot_loss_list.append(running_rot_loss/200)
            trans_loss_list.append(running_trans_loss/200)
            running_loss = 0
            running_rot_loss = 0
            running_trans_loss = 0
x = []
for i in range(50):
    x.append(200 * (i + 1))
plt.figure(figsize=(8, 4))
plt.plot(x, loss_list, label="$loss$", color="red", linewidth=2)
plt.xlabel("training data pairs")
plt.ylabel("loss")
plt.show()
plt.savefig("loss.jpg")

plt.figure(figsize=(8, 4))
plt.plot(x, rot_loss_list, label="$rot_loss$", color="red", linewidth=2)
plt.xlabel("training data pairs")
plt.ylabel("rot_loss")
plt.show()
plt.savefig("rot_loss.jpg")

plt.figure(figsize=(8, 4))
plt.plot(x, trans_loss_list, label="$trans_loss$", color="red", linewidth=2)
plt.xlabel("training data pairs")
plt.ylabel("trans_loss")
plt.show()
plt.savefig("trans_loss.jpg")
# torch.save(resnet18, 'model.pkl')
if torch.cuda.is_available():
    print("cuda is available!\n")
    device = torch.device("cuda")
    x = torch.randn(4, 4)
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))

# [1,   200] loss: 0.933
# [1,   400] loss: 0.590
# [1,   600] loss: 0.552
# [1,   800] loss: 0.562
# [1,  1000] loss: 0.602
# [2,   200] loss: 0.550
# [2,   400] loss: 0.520
# [2,   600] loss: 0.510
# [2,   800] loss: 0.464
# [2,  1000] loss: 0.405
# [3,   200] loss: 0.393
# [3,   400] loss: 0.330
# [3,   600] loss: 0.368
# [3,   800] loss: 0.333
# [3,  1000] loss: 0.309
# [4,   200] loss: 0.305
# [4,   400] loss: 0.318
# [4,   600] loss: 0.293
# [4,   800] loss: 0.293
# [4,  1000] loss: 0.270
# [5,   200] loss: 0.258
# [5,   400] loss: 0.242
# [5,   600] loss: 0.250
# [5,   800] loss: 0.220
# [5,  1000] loss: 0.213
# [6,   200] loss: 0.216
# [6,   400] loss: 0.191
# [6,   600] loss: 0.198
# [6,   800] loss: 0.191
# [6,  1000] loss: 0.198
# [7,   200] loss: 0.168
# [7,   400] loss: 0.181
# [7,   600] loss: 0.179
# [7,   800] loss: 0.171
# [7,  1000] loss: 0.169
# [8,   200] loss: 0.170
# [8,   400] loss: 0.157
# [8,   600] loss: 0.159
# [8,   800] loss: 0.138
# [8,  1000] loss: 0.131
# [9,   200] loss: 0.145
# [9,   400] loss: 0.129
# [9,   600] loss: 0.132
# [9,   800] loss: 0.137
# [9,  1000] loss: 0.133
# [10,   200] loss: 0.134
# [10,   400] loss: 0.129
# [10,   600] loss: 0.123
# [10,   800] loss: 0.141
# [10,  1000] loss: 0.129

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

preprocess = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
# step 1: load the trained model
model = torch.load('model.pkl')
# step 2: set the model into evaluation mode and disable all the change
model.eval()
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    model.cuda()
# step 3: prepare the dataset, dataLoader and loss function
test_dataset = RGBDDataset('office', './', 'seq-02', 1000)
dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
loss_func = torch.nn.L1Loss()
running_loss = torch.zeros(1)
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

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda:0')
        pose_data = pose_data.to('cuda:0')
    prediction = model(input_batch)
    # print(prediction.double().view(16))
    # print(frame_pose_input)

    loss = loss_func(prediction.double().view(6), pose_data).cuda()

    running_loss += loss
    if i % 50 == 49:
        print('[%5d] loss: %.3f' % (i + 1, running_loss / 50))
        running_loss = 0
# [   50] loss: 0.464
# [  100] loss: 0.444
# [  150] loss: 0.426
# [  200] loss: 0.428
# [  250] loss: 0.460
# [  300] loss: 0.441
# [  350] loss: 0.456
# [  400] loss: 0.497
# [  450] loss: 0.433
# [  500] loss: 0.471
# [  550] loss: 0.434
# [  600] loss: 0.460
# [  650] loss: 0.458
# [  700] loss: 0.499
# [  750] loss: 0.458
# [  800] loss: 0.494
# [  850] loss: 0.423
# [  900] loss: 0.445
# [  950] loss: 0.462
# [ 1000] loss: 0.444

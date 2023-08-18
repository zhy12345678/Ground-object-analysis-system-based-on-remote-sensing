from tensorflow.keras.models import load_model
import numpy
import socket
import _thread
from model import get_model
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
from torchvision import transforms, datasets, utils,models
import random
import string
import time
import matplotlib.pyplot as plt



def predictvgg16_7(image_path):
    model = load_model('weightsVGG16_7.h5')
    model.summary()
    image_path
    img=Image.open(image_path)
    img=img.resize((224,224),Image.ANTIALIAS)
    img=numpy.array(img)
    img_tensor=numpy.expand_dims(img,axis=0)
    img_tensor=img_tensor/255.
    print('该图像的尺寸为：', img_tensor.shape)
    time_start = time.time()
    prediction = model(img_tensor)
    time_end = time.time()
    time_sum = time_end - time_start
    time_last = ('%.2f' % time_sum)
    labels = {0: 'Field', 1: 'Forest', 2: 'Grass', 3: 'Industry', 4: 'Parking', 5: 'Resident',
              6: 'RiverLake'}
    pre_y = labels[numpy.argmax(prediction)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
    ax[0].set_title(pre_y)
    ax[0].imshow(img)
    ax[1].bar(x=list(labels.values()), height=prediction.numpy().ravel())
    for x, y in zip(list(labels.values()), prediction.numpy().ravel()):
        ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
    ax[1].set_ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    road =ran_str + '.jpg'
    plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
    print(pre_y)
    print(time_last)
    print(road)
    last=pre_y+"/#/"+time_last+"/#/"+road
    return last


def predictvgg16_11(image_path):
    model = load_model('weightsVGG16_11.h5')
    model.summary()
    image_path
    img=Image.open(image_path)
    img=img.resize((224,224),Image.ANTIALIAS)
    img=numpy.array(img)
    img_tensor=numpy.expand_dims(img,axis=0)
    img_tensor=img_tensor/255.
    print('该图像的尺寸为：', img_tensor.shape)
    time_start = time.time()
    prediction = model(img_tensor)
    time_end = time.time()
    time_sum = time_end - time_start
    time_last = ('%.2f' % time_sum)
    labels = {0: 'denseforest', 1: 'grassland', 2: 'harbor', 3: 'highbuildings', 4: 'lowbuildings', 5: 'overpass',
              6: 'railway', 7: 'residentialarea', 8: 'roads', 9: 'sparseforest', 10: 'storagetanks'}
    pre_y = labels[numpy.argmax(prediction)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
    ax[0].set_title(pre_y)
    ax[0].imshow(img)
    ax[1].bar(x=list(labels.values()), height=prediction.numpy().ravel())
    for x, y in zip(list(labels.values()), prediction.numpy().ravel()):
        ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
    ax[1].set_ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    road = ran_str + '.jpg'
    plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
    print(pre_y)
    print(time_last)
    print(road)
    last = pre_y + "/#/" + time_last + "/#/" + road
    return last


def predictvgg16_12(image_path):
    model = load_model('weightsVGG16_12.h5')
    model.summary()
    image_path
    img=Image.open(image_path)
    img=img.resize((224,224),Image.ANTIALIAS)
    img=numpy.array(img)
    img_tensor=numpy.expand_dims(img,axis=0)
    img_tensor=img_tensor/255.
    print('该图像的尺寸为：', img_tensor.shape)
    time_start = time.time()
    prediction = model(img_tensor)
    time_end = time.time()
    time_sum = time_end - time_start
    time_last = ('%.2f' % time_sum)
    labels = {0: 'agriculture', 1: 'commercial', 2: 'harbor', 3: 'idle_land', 4: 'industrial', 5: 'meadow',
              6: 'overpass', 7: 'park', 8: 'pond', 9: 'residential', 10: 'river', 11: 'water'}
    pre_y = labels[numpy.argmax(prediction)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
    ax[0].set_title(pre_y)
    ax[0].imshow(img)
    ax[1].bar(x=list(labels.values()), height=prediction.numpy().ravel())
    for x, y in zip(list(labels.values()), prediction.numpy().ravel()):
        ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
    ax[1].set_ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    road = ran_str + '.jpg'
    plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
    print(pre_y)
    print(time_last)
    print(road)
    last = pre_y + "/#/" + time_last + "/#/" + road
    return last

#######################################################################################
def predictvgg19_7(image_path):
    model = load_model('weightsVGG19_7.h5')
    model.summary()
    image_path
    img=Image.open(image_path)
    img=img.resize((224,224),Image.ANTIALIAS)
    img=numpy.array(img)
    img_tensor=numpy.expand_dims(img,axis=0)
    img_tensor=img_tensor/255.
    print('该图像的尺寸为：', img_tensor.shape)
    time_start = time.time()
    prediction = model(img_tensor)
    time_end = time.time()
    time_sum = time_end - time_start
    time_last = ('%.2f' % time_sum)
    labels = {0: 'Field', 1: 'Forest', 2: 'Grass', 3: 'Industry', 4: 'Parking', 5: 'Resident',
              6: 'RiverLake'}
    pre_y = labels[numpy.argmax(prediction)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
    ax[0].set_title(pre_y)
    ax[0].imshow(img)
    ax[1].bar(x=list(labels.values()), height=prediction.numpy().ravel())
    for x, y in zip(list(labels.values()), prediction.numpy().ravel()):
        ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
    ax[1].set_ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    road =ran_str + '.jpg'
    plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
    print(pre_y)
    print(time_last)
    print(road)
    last = pre_y + "/#/" + time_last + "/#/" + road
    return last

#######################################################################################################
def predictvgg19_11(image_path):
    model = load_model('weightsVGG19_11.h5')
    model.summary()
    image_path
    img=Image.open(image_path)
    img=img.resize((224,224),Image.ANTIALIAS)
    img=numpy.array(img)
    img_tensor=numpy.expand_dims(img,axis=0)
    img_tensor=img_tensor/255.
    print('该图像的尺寸为：', img_tensor.shape)
    time_start = time.time()
    prediction = model(img_tensor)
    time_end = time.time()
    time_sum = time_end - time_start
    time_last = ('%.2f' % time_sum)
    labels = {0: 'denseforest', 1: 'grassland', 2: 'harbor', 3: 'highbuildings', 4: 'lowbuildings', 5: 'overpass',
              6: 'railway', 7: 'residentialarea', 8: 'roads', 9: 'sparseforest', 10: 'storagetanks'}
    pre_y = labels[numpy.argmax(prediction)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
    ax[0].set_title(pre_y)
    ax[0].imshow(img)
    ax[1].bar(x=list(labels.values()), height=prediction.numpy().ravel())
    for x, y in zip(list(labels.values()), prediction.numpy().ravel()):
        ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
    ax[1].set_ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    road = ran_str + '.jpg'
    plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
    print(pre_y)
    print(time_last)
    print(road)
    last = pre_y + "/#/" + time_last + "/#/" + road
    return last
###################################################################################################################3
def predictvgg19_12(image_path):
    model = load_model('weightsVGG19_12.h5')
    model.summary()
    image_path
    img=Image.open(image_path)
    img=img.resize((224,224),Image.ANTIALIAS)
    img=numpy.array(img)
    img_tensor=numpy.expand_dims(img,axis=0)
    img_tensor=img_tensor/255.
    print('该图像的尺寸为：', img_tensor.shape)
    time_start = time.time()
    prediction = model(img_tensor)
    time_end = time.time()
    time_sum = time_end - time_start
    time_last = ('%.2f' % time_sum)
    labels = {0: 'agriculture', 1: 'commercial', 2: 'harbor', 3: 'idle_land', 4: 'industrial', 5: 'meadow',
              6: 'overpass', 7: 'park', 8: 'pond', 9: 'residential', 10: 'river', 11: 'water'}
    pre_y = labels[numpy.argmax(prediction)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
    ax[0].set_title(pre_y)
    ax[0].imshow(img)
    ax[1].bar(x=list(labels.values()), height=prediction.numpy().ravel())
    for x, y in zip(list(labels.values()), prediction.numpy().ravel()):
        ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
    ax[1].set_ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    road = ran_str + '.jpg'
    plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
    print(pre_y)
    print(time_last)
    print(road)
    last = pre_y + "/#/" + time_last + "/#/" + road
    return last

####################################################################################

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size = None):
        super().__init__()
        size = size or (1,1) #池化层的卷积核大小，默认为(1,1)
        self.pool_one = nn.AdaptiveAvgPool2d(size)#池化层1
        self.pool_two = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        return torch.cat([self.pool_one(x), self.pool_two(x)],1)

#迁移学习：得到一个成熟的模型，进行模型微调
######################################################################
def get_model7():
    model_pre = models.resnet50(pretrained= True)#获取与训练模型
    for param in model_pre.parameters():
        param.requires_grad = False

    #替换resnet最后两层网路，返回一个新的模型
    model_pre.avgpool = AdaptiveConcatPool2d()#替换池化层
    model_pre.fc = nn.Sequential(
        nn.Flatten(),#拉平所有维度
        nn.BatchNorm1d(4096),#256*6*6  -->4096
        nn.Dropout(0.5),
        nn.Linear(4096,512),
        nn.ReLU(),
        nn.BatchNorm1d(512),#正则化处理
        nn.Dropout(0.5),
        nn.Linear(512,7),
        nn.LogSoftmax(dim=1)#损失函数
     )
    return model_pre


def predictresnet50_7(image_path):
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   data_transform = transforms.Compose(
       [transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
   class_indict = {"0": "Field","1": "Forest","2": "Grass","3": "Industry","4": "Parking","5": "Resident","6": "RiverLake"}
   model = get_model7()
   model_weight_path ='weightsResNet50_7.pth'
   model.load_state_dict(torch.load(model_weight_path, map_location=device))
   img = Image.open(image_path).convert('RGB')
   img = data_transform(img)
   img = torch.unsqueeze(img, dim=0)
   model.eval()

   with torch.no_grad():
       time_start = time.time()
       output = torch.squeeze(model(img))
       predict = torch.softmax(output, dim=0)
       predict_cla = torch.argmax(predict).numpy()
       time_end = time.time()
       time_sum = time_end - time_start
       time_last = ('%.2f' % time_sum)
       fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
       ax[0].set_title(class_indict[str(predict_cla)])
       img = Image.open(image_path).convert('RGB')
       ax[0].imshow(img)
       ax[1].bar(x=list(class_indict.values()), height=predict.numpy().ravel())
       for x, y in zip(list(class_indict.values()), predict.numpy().ravel()):
           ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
       ax[1].set_ylim(0, 1.1)
       plt.xticks(rotation=90)
       plt.tight_layout()
       ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
       road = ran_str + '.jpg'
       plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
       print(class_indict[str(predict_cla)])
       print(time_last)
       print(road)
   last = class_indict[str(predict_cla)] + "/#/" + time_last + "/#/" + road
   return last

#####################################################################################
def get_model11():
    model_pre = models.resnet50(pretrained= True)#获取与训练模型
    for param in model_pre.parameters():
        param.requires_grad = False
    #替换resnet最后两层网路，返回一个新的模型
    model_pre.avgpool = AdaptiveConcatPool2d()#替换池化层
    model_pre.fc = nn.Sequential(
        nn.Flatten(),#拉平所有维度
        nn.BatchNorm1d(4096),#256*6*6  -->4096
        nn.Dropout(0.5),
        nn.Linear(4096,512),
        nn.ReLU(),
        nn.BatchNorm1d(512),#正则化处理
        nn.Dropout(0.5),
        nn.Linear(512,11),
        nn.LogSoftmax(dim=1)#损失函数
     )
    return model_pre


def predictresnet50_11(image_path):
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   data_transform = transforms.Compose(
       [transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
   class_indict = {
    "0": "denseforest",
    "1": "grassland",
    "2": "harbor",
    "3": "highbuildings",
    "4": "lowbuildings",
    "5": "overpass",
    "6": "railway",
    "7": "residentialarea",
    "8": "roads",
    "9": "sparseforest",
    "10": "stroagetanks"
}
   model = get_model11()
   model_weight_path = 'weightsResnet50_11.pth'
   model.load_state_dict(torch.load(model_weight_path, map_location=device))
   img = Image.open(image_path).convert('RGB')
   img = data_transform(img)
   img = torch.unsqueeze(img, dim=0)
   model.eval()
   with torch.no_grad():
       time_start = time.time()
       output = torch.squeeze(model(img))
       predict = torch.softmax(output, dim=0)
       predict_cla = torch.argmax(predict).numpy()
       time_end = time.time()
       time_sum = time_end - time_start
       time_last = ('%.2f' % time_sum)
       fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
       ax[0].set_title(class_indict[str(predict_cla)])
       img = Image.open(image_path).convert('RGB')
       ax[0].imshow(img)
       ax[1].bar(x=list(class_indict.values()), height=predict.numpy().ravel())
       for x, y in zip(list(class_indict.values()), predict.numpy().ravel()):
           ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
       ax[1].set_ylim(0, 1.1)
       plt.xticks(rotation=90)
       plt.tight_layout()
       ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
       road = ran_str + '.jpg'
       plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
       print(class_indict[str(predict_cla)])
       print(time_last)
       print(road)
   last = class_indict[str(predict_cla)] + "/#/" + time_last + "/#/" + road
   return last

#############################################################################
def get_model12():
    model_pre = models.resnet50(pretrained= True)#获取与训练模型
    for param in model_pre.parameters():
        param.requires_grad = False

    #替换resnet最后两层网路，返回一个新的模型
    model_pre.avgpool = AdaptiveConcatPool2d()#替换池化层
    model_pre.fc = nn.Sequential(
        nn.Flatten(),#拉平所有维度
        nn.BatchNorm1d(4096),#256*6*6  -->4096
        nn.Dropout(0.5),
        nn.Linear(4096,512),
        nn.ReLU(),
        nn.BatchNorm1d(512),#正则化处理
        nn.Dropout(0.5),
        nn.Linear(512,12),
        nn.LogSoftmax(dim=1)#损失函数
     )
    return model_pre


def predictresnet50_12(image_path):
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   data_transform = transforms.Compose(
       [transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
   class_indict = {"0": "agriculture","1": "commercial","2": "harbor","3": "idle_land","4": "industrial","5": "meadow","6": "overpass","7": "park",
    "8": "pond","9": "residential","10": "river","11": "water"}
   model = get_model12()
   model_weight_path = 'weightsResnet50_12.pth'
   model.load_state_dict(torch.load(model_weight_path, map_location=device))
   img = Image.open(image_path).convert('RGB')
   img = data_transform(img)
   img = torch.unsqueeze(img, dim=0)
   model.eval()
   with torch.no_grad():
       time_start = time.time()
       output = torch.squeeze(model(img))
       predict = torch.softmax(output, dim=0)
       predict_cla = torch.argmax(predict).numpy()
       time_end = time.time()
       time_sum = time_end - time_start
       time_last = ('%.2f' % time_sum)
       fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi=70)
       ax[0].set_title(class_indict[str(predict_cla)])
       img = Image.open(image_path).convert('RGB')
       ax[0].imshow(img)
       ax[1].bar(x=list(class_indict.values()), height=predict.numpy().ravel())
       for x, y in zip(list(class_indict.values()), predict.numpy().ravel()):
           ax[1].text(x=x, y=y + 0.02, s='{:.2f}'.format(y), horizontalalignment='center')
       ax[1].set_ylim(0, 1.1)
       plt.xticks(rotation=90)
       plt.tight_layout()
       ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
       road = ran_str + '.jpg'
       plt.savefig('D:\Material1' + '\\' + ran_str + '.jpg')
       print(class_indict[str(predict_cla)])
       print(time_last)
       print(road)

   last = class_indict[str(predict_cla)] + "/#/" + time_last + "/#/" + road
   return last

##########################################################################
def read_msg(sk):
        print("到了")
        data = sk.recv(1024).decode('utf-8').split('/#/')
        data = [i.replace('\r\n', '') for i in data]
        print('来自Java的信息: ', data)
        if data[1]=='VGG16' and data[2]=='RSSCN7':
            sk.send((predictvgg16_7(data[0]) + '\r\n').encode('utf-8'))
        elif data[1]=='VGG16' and data[2]=='RS_C11_Database':
            sk.send((predictvgg16_11(data[0]) + '\r\n').encode('utf-8'))
        elif data[1]=='VGG16' and data[2]=='SIRI WHU':
            sk.send((predictvgg16_12(data[0]) + '\r\n').encode('utf-8'))
        elif data[1]=='VGG19' and data[2]=='RSSCN7':
            sk.send((predictvgg19_7(data[0]) + '\r\n').encode('utf-8'))
        elif data[1]=='VGG19' and data[2]=='RS_C11_Database':
            sk.send((predictvgg19_11(data[0]) + '\r\n').encode('utf-8'))
        elif data[1]=='VGG19' and data[2]=='SIRI WHU':
            sk.send((predictvgg19_12(data[0]) + '\r\n').encode('utf-8'))
        elif data[1] == 'ResNet50' and data[2] == 'RSSCN7':
            sk.send((predictresnet50_7(data[0]) + '\r\n').encode('utf-8'))
        elif data[1] == 'ResNet50' and data[2] == 'RS_C11_Database':
            sk.send((predictresnet50_11(data[0]) + '\r\n').encode('utf-8'))
        elif data[1] == 'ResNet50' and data[2] == 'SIRI WHU':
            sk.send((predictresnet50_12(data[0]) + '\r\n').encode('utf-8'))
        sk.close()
if __name__ == '__main__':
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    addr = 'localhost'
    port = 9529
    sk.bind((addr, port))
    sk.listen(1)
    while True:
        clientsocket, addr = sk.accept()
        _thread.start_new_thread(read_msg, (clientsocket,))



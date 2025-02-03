#yolov5
import cv2
import torch
#C:\Users\admin\.cache\torch\hub\ultralytics_yolov5_master
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 读取图片
img = cv2.imread('street.jpg')

# 进行推理
results = model(img)

# 获取检测结果的图像
output_img = cv2.resize(results.render()[0],(512,512))
print(output_img.shape)

# 显示图像
cv2.imshow('YOLOv5', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Unet
""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义DoubleConv模块，包含两个卷积层、批归一化和ReLU激活函数
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 定义双卷积层序列
        self.double_conv = nn.Sequential(
            # 第一个卷积层，使用3x3卷积核，填充为1，保持特征图尺寸不变
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 批归一化层，加速收敛
            nn.BatchNorm2d(out_channels),
            # ReLU激活函数，使用inplace=True节省内存
            nn.ReLU(inplace=True),
            # 第二个卷积层，同样使用3x3卷积核，填充为1
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播，将输入通过双卷积层序列
        return self.double_conv(x)


# 定义下采样模块，包含一个最大池化层和一个DoubleConv模块
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        # 定义最大池化和双卷积层序列
        self.maxpool_conv = nn.Sequential(
            # 最大池化层，池化核大小为2，步长为2，进行下采样
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # 前向传播，将输入通过最大池化和双卷积层序列
        return self.maxpool_conv(x)


# 定义上采样模块，根据是否使用双线性插值选择不同的上采样方式，然后进行双卷积
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # 如果使用双线性插值进行上采样
        if bilinear:
            # 定义上采样层，使用双线性插值，缩放因子为2
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 否则使用反卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # 定义双卷积层
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 对输入x1进行上采样
        x1 = self.up(x1)
        # 计算x1和x2在高度和宽度上的差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行填充，使其与x2在高度和宽度上一致
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 将x2和填充后的x1在通道维度上拼接
        x = torch.cat([x2, x1], dim=1)
        # 将拼接后的结果通过双卷积层
        return self.conv(x)


# 定义输出卷积模块，使用1x1卷积核
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 定义1x1卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 前向传播，将输入通过1x1卷积层
        return self.conv(x)


# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        # 输入通道数
        self.n_channels = n_channels
        # 输出类别数
        self.n_classes = n_classes
        # 是否使用双线性插值进行上采样
        self.bilinear = bilinear

        # 定义输入的双卷积层
        self.inc = DoubleConv(n_channels, 64)
        # 定义下采样模块
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        # 定义上采样模块
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # 定义输出卷积层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 输入通过输入的双卷积层
        x1 = self.inc(x)
        # 进行下采样
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 进行上采样，同时与下采样对应的特征图进行拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # 输出卷积，得到最终的logits
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # 创建U-Net模型实例，输入通道数为3，输出类别数为1
    net = UNet(n_channels=3, n_classes=1)
    # 打印模型结构
    print(net)


#Deepsort
import cv2
import torch
from deep_sort import DeepSort
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords

# 加载YOLOv5模型
model = attempt_load('yolov5s.pt', map_location='cpu')

# 初始化DeepSORT
deepsort = DeepSort("deep_sort/mars-small128.pb")

# 打开视频文件
cap = cv2.VideoCapture('test5.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5目标检测
    img = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)

    # 处理检测结果
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            bboxes = det[:, :4].cpu().numpy()
            confidences = det[:, 4].cpu().numpy()
            class_ids = det[:, 5].cpu().numpy()

            # DeepSORT目标跟踪
            tracks = deepsort.update(bboxes, confidences, class_ids, frame)

            # 绘制跟踪结果
            for track in tracks:
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Openpose
import cv2
import sys
import os

# 添加 OpenPose 的 Python 路径
sys.path.append("./openpose/python")  # 替换为你的 openpose/python 路径
from openpose import pyopenpose as op

# 设置 OpenPose 参数
params = {
    "model_folder": "./openpose/models",  # 替换为你的 models 路径
    "net_resolution": "368x368",  # 网络输入分辨率
    "number_people_max": 1,  # 最多检测的人数
    "render_threshold": 0.05,  # 渲染阈值
    "disable_blending": False  # 是否禁用混合
}

# 初始化 OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 读取图片
image_path = "demo.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)

# 处理图片
datum = op.Datum()
datum.cvInputData = image
opWrapper.emplaceAndPop([datum])

# 获取结果
keypoints = datum.poseKeypoints  # 人体关键点坐标
output_image = datum.cvOutputData  # 渲染后的图片

# 显示结果
print("Body keypoints: ", keypoints)
cv2.imshow("OpenPose Result", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


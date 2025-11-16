import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary


# バッチサイズ
BATCH_SIZE = 8

# データ集合の準備
def create_dataset():
    # 訓練データ
    train_data=CIFAR10("./data",train=True,transform=ToTensor(),download=True)
    # テストデータ
    test_data=CIFAR10("./data",train=False,transform=ToTensor(),download=True)

    return train_data,test_data

# CIFAR10データセットは(3,32,32)の画像。
# 畳み込み層、プーリング層の出力した特徴マップのサイズは(（W + 2P - F）//S) + 1となる
# 今度はパディングを実施しなため、Pは0

# CNNの構築
class ImageClassificationModel(nn.Module):
    # 親クラスの初期化
    def __init__(self):
        super().__init__()
        # 1つ目の畳み込み層、入力のチャネルは3(R,G,B)、出力のチャネルは6、フィルタサイズ3、ストライド1、パディングなし
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
        # 1つ目のプーリング層、フィルタサイズ2、ストライド2、パディングなし→特徴マップのサイズを半減する
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 2つ目の畳み込み層、入力のチャネルは6(conv1のoutputが6だから)、出力のチャネル16、フィルタサイズ3、ストライド1、パディングなし
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0
        )
        # 2つ目のプーリング層、フィルタサイズ2、ストライド2、パディングなし→特徴マップのサイズを半減する
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 1つ目のアフィン層、576は16 * 6 * 6からなる(pool2がconv2から16個の特徴マップをもらったのサイズ15から6へ縮小したため)
        self.linear1=nn.Linear(576,120)
        # 2つ目のアフィン層
        self.linear2 = nn.Linear(120, 84)
        # 出力層、10個の分類のため出力は10
        self.output = nn.Linear(84, 10)

    # フォワードプロパゲーション
    def forward(self,x):
        # 第1層: 畳み込み → 活性化関数(RELU) → プーリング
        x=self.pool1(torch.relu(self.conv1(x)))

        # 第2層: 畳み込み → 活性化関数(RELU) → プーリング
        x = self.pool2(torch.relu(self.conv2(x)))
        
        # アフィン層は2次元のテンソルしか処理できないため、第2層の出力データを2次元に変換する必要がある
        x=x.view(
            x.size(0),
            -1
        )
        print("flatten shape:", x.shape)
        # 第3層: 1つ目のアフィン層
        x=torch.relu(self.linear1(x))

        # 第4層: 2つ目のアフィン層
        x=torch.relu(self.linear2(x))

        # 第5層: 出力層
        return self.output(x)






if __name__=="__main__":
    # モデルのインスタンス化
    model=ImageClassificationModel()
    # モデルのパラメータを見る
    summary(model=model,input_size=(3,32,32),batch_size=BATCH_SIZE)

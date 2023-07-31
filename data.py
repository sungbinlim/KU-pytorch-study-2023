import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import pandas as pd
from io import BytesIO
from PIL import Image # Bytes 로 압축된 이미지를 이미지로 변환하기 위해 필요
from zipfile import ZipFile # 압축된 데이터 파일 핸들링을 위한 라이브러리

class IonDataset(Dataset):
    def __init__(self, data_dir, mode='train', img_size=(10, 300), n=None):
        self.df = pd.read_pickle(os.path.join(data_dir, f"{mode}.pkl"))

        if n is not None:  # to extract a specific n
            self.df = self.df[self.df["n"] == n]

        self.data_dir = data_dir
        self.image_path = self.df["img_path"].values

        self.min_a2 = 0.146
        self.max_a2 = 0.522
        
        self.df["norm_potential"] = self.df["potential"].apply(lambda p : (p - self.min_a2)/ (self.max_a2 - self.min_a2))  # min-max normalization

        self.n = self.df["n"].values
        self.max_n = self.n.max()
        self.min_n = self.n.min()
        
        self.potential = self.df["potential"].values.astype('float32')
        self.norm_potential = self.df["norm_potential"].values.astype('float32')
        self.position = self.df["position"].values

        self.h, self.w = img_size
        self.crop_size = (10, self.w)
        self.img_transform = T.Compose([
            T.ToTensor(),
            # T.CenterCrop(self.crop_size),
        ])

    def make_one_hot(self, scalar):
        one_hot_dim = self.max_n - self.min_n + 1
        address = scalar - one_hot_dim - 1
        one_hot_vec = np.zeros(one_hot_dim)
        one_hot_vec[address] = 1.0
        return one_hot_vec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.data_dir, self.image_path[idx]))
        img = self.img_transform(img)
        ion_number = torch.tensor(self.make_one_hot(self.n[idx]))
        ion_potential = torch.tensor(self.norm_potential[idx])
        
        # n - 1 to use cross entropy
        return img, ion_number, ion_potential
    
def extract_img_zipfile(zip_path, img_name, idx):
    """
    압축파일에서 사진파일을 읽는 함수
    zip_path: zip 파일 경로 ex) './train.zip'
    img_name: zip 파일 내 이미지 파일 이름 ex) 'train/image_0.png' or ['train/image_0.png', 'train/image_1.png', ..]
    """
    img_data = []
    with ZipFile(zip_path, 'r') as img_file: 
        # zip_path 경로 내 img_name 데이터를 읽는 코드
        img_data = [Image.open(BytesIO(img_file.read(img_name[id]))) for id in idx]
    
    # io.BytesIO 로 Byte 객체 생성
    imgs = [np.array(img) for img in img_data]
    return np.array(imgs).reshape(len(idx), imgs[0].shape[0], imgs[0].shape[1])


import h5py
import os
from PIL import Image
from torchvision import transforms
import config
import numpy
from tqdm import tqdm
import pickle

transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def image_preprocessing(images_path, image_size, processed_image_path, img2idx_path):
    num_of_pics = len(os.listdir(images_path))

    features_shape = (num_of_pics, 3, image_size[0], image_size[1])
    img2idx = {}
    with h5py.File(processed_image_path, 'w', libver='latest') as f:
        images = f.create_dataset('images', shape=features_shape, dtype='float16')

        i = 0
        for image_name in tqdm(os.listdir(images_path)):
            image_path = os.path.join(images_path, image_name)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            images[i, :, :] = image.numpy().astype('float16')
            #id_and_extension = image_name
            id = int(image_name.split('.')[0])
            img2idx[id] = i
            i += 1

    with open(img2idx_path, 'wb') as f:
        pickle.dump(img2idx, f)

def image_preprocessing_master():
    image_preprocessing('data/CelebA/img_align_celeba', (64,64), 'data/cache/train.h5', 'data/cache/img2idx_train.pkl')

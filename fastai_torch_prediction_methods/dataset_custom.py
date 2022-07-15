import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize


class ImageDataset(Dataset):
    def __init__(self, file_paths, img_size):
        self.file_paths = file_paths
        self.img_size = img_size

    def __getitem__(self, index):
        fpath = self.file_paths[index]
        pil_image = Image.open(fpath)
        resize = T.Resize([self.img_size, self.img_size])
        res_pil_image = resize(pil_image)
        img = T.ToTensor()(res_pil_image)
        return img

    def __len__(self):
        return len(self.file_paths)

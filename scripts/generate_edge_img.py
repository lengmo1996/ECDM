import cv2, torch, os
import numpy as np
from pathlib import Path
from glob import glob

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes

llvip_dataset_path = "data/LLVIP"
prw_dataset_path = "data/PRW-v16.04.20"

image_folder_list = [
    Path(llvip_dataset_path) / "infrared",
    Path(llvip_dataset_path) / "visible",
    Path(prw_dataset_path) / "frame",
]

for image_folder in image_folder_list:
    root_path = glob(str(Path(image_folder) / "**" / "*.*"), recursive=True)
    image_path = sorted(
        x.replace("/", os.sep)
        for x in root_path
        if x.split(".")[-1].lower() in IMG_FORMATS
    )
    for path in image_path:
        img = cv2.imread(str(path), 0)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[crow - 15 : crow + 15, ccol - 15 : ccol + 15] = 0

        ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)
        iimg = torch.from_numpy(iimg).type(torch.uint8).numpy()

        split_ = str(path).split("/")

        split_[3] = split_[3] + "_HF"
        path_HF = Path("/".join(split_))
        os.makedirs(path_HF.parent, exist_ok=True)
        cv2.imwrite(str(path_HF), iimg)
        print(path_HF)

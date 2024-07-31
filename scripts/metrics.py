from cleanfid import fid
from pathlib import Path
from cleanfid import fid
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import lpips
from glob import glob
import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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


class PairDataset(Dataset):
    def __init__(self, dataset1, dataset2) -> None:
        self.dataset1 = Path(dataset1)

        self.dataset1 = glob(str(Path(self.dataset1) / "**" / "*.*"), recursive=True)
        self.dataset1 = sorted(
            x.replace("/", os.sep)
            for x in self.dataset1
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.dataset2 = Path(dataset2)
        self.dataset2 = glob(str(Path(self.dataset2) / "**" / "*.*"), recursive=True)
        self.dataset2 = sorted(
            x.replace("/", os.sep)
            for x in self.dataset2
            if x.split(".")[-1].lower() in IMG_FORMATS
        )

        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)
        assert self.len1 == self.len2, "unpaired datasets"
        self.transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def __len__(self):
        return self.len1

    def __getitem__(self, index):
        img1_path = self.dataset1[index]
        img2_path = self.dataset2[index]

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        img1 = img1.convert("L")
        img2 = img2.convert("L")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def get_lpips_ssim_metrics(dataset1, dataset2, log_file, batch_size=64):
    dataset = PairDataset(dataset1, dataset2)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size
    )

    lpips_metric = lpips.LPIPS(net="alex")
    ddim_metric = lpips.DSSIM(colorspace="RGB")
    total_num = len(dataset)

    init_lpips = 0
    init_ssim = 0
    init_psnr = 0

    with torch.no_grad():
        for i, (img1, img2) in tqdm(
            enumerate(dataloader),
            desc="LPIPS",
            initial=0,
            total=int(total_num / batch_size),
        ):
            img1, img2 = img1.cuda(), img2.cuda()
            lpips_metric = lpips_metric.cuda()
            d_lpips = lpips_metric.forward(img1, img2)
            init_lpips = init_lpips + d_lpips.sum()

    final_lpips = init_lpips / (total_num + 1e-8)
    with open(log_file, "a") as f:
        f.write("lpips:{0}".format(final_lpips))

    dataloader2 = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=1
    )
    with torch.no_grad():
        for i, (img1, img2) in tqdm(
            enumerate(dataloader2), desc="SSIM+PSNR", initial=0, total=total_num
        ):
            img1, img2 = img1.cuda(), img2.cuda()
            ddim_metric = ddim_metric.cuda()
            d_ssim = ddim_metric.forward(img1, img2)
            init_ssim = init_ssim + d_ssim

            d_psnr = psnr(img1, img2)
            init_psnr = init_psnr + d_psnr

    final_dssim = init_ssim / (total_num + 1e-8)
    final_psnr = init_psnr / (total_num + 1e-8)

    with open(log_file, "a") as f:
        f.write(
            "dssim:{0},ssim:{1},psnr:{2}\n".format(
                final_dssim, 1.0 - 2.0 * (final_dssim), final_psnr
            )
        )


def write_log(
    dataset1,
    dataset2,
    log_file,
    fid_score_pytorch_v3,
    fid_score_clean_v3,
    fid_score_clean_clip,
    kid_score_pytorch_v3,
):
    with open(log_file, "a") as f:
        f.write("first_dataset:{0},second_daatset:{1}\n".format(dataset1, dataset2))
        f.write(
            "fid_score_pytorch_v3:{0},fid_score_clean_v3:{1},fid_score_clean_clip:{2},kid_score_pytorch_v3:{3}\n".format(
                fid_score_pytorch_v3,
                fid_score_clean_v3,
                fid_score_clean_clip,
                kid_score_pytorch_v3,
            )
        )


def get_all_fid(dataset1, dataset2, log_file):
    fid_score_clean_v3 = fid.compute_fid(
        dataset1,
        dataset2,
        mode="clean",
        model_name="inception_v3",
        num_workers=64,
        batch_size=64,
    )
    fid_score_pytorch_v3 = fid.compute_fid(
        dataset1,
        dataset2,
        mode="legacy_pytorch",
        model_name="inception_v3",
        num_workers=64,
        batch_size=64,
    )
    fid_score_clean_clip = fid.compute_fid(
        dataset1,
        dataset2,
        mode="clean",
        model_name="clip_vit_b_32",
        num_workers=64,
        batch_size=64,
    )
    kid_score_pytorch_v3 = fid.compute_kid(
        dataset1, dataset2, mode="clean", num_workers=64, batch_size=64
    )
    write_log(
        dataset1,
        dataset2,
        log_file,
        fid_score_pytorch_v3,
        fid_score_clean_v3,
        fid_score_clean_clip,
        kid_score_pytorch_v3,
    )


if __name__ == "__main__":
    log_file = "metric_lpips_psnr.txt"
    log_file = "metric_fid.txt"
    real_path = "data/LLVIP/infrared/test"
    generated_path = "data/LLVIP/infrared/test"
    get_all_fid(real_path, generated_path, log_file)
    get_lpips_ssim_metrics(real_path, generated_path, log_file)

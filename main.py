### Main entry point for the project 

from utils import download_url
from model import VGG 
import torch


def main():

    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    model = VGG().cuda()
    print(f"=> loading checkpoint '{checkpoint_url}'")
    model.load_state_dict(checkpoint['state_dict'])
    recover_model = lambda : model.load_state_dict(checkpoint['state_dict'])


if __name__ == "__main__":
    main()

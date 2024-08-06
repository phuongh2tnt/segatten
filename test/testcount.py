import sys
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
import utils.metric2 as metrics
from utils.iris_dataset import visualize
from PIL import Image, ImageOps, ImageDraw, ImageFont
from timeit import default_timer as timer
from scipy.ndimage import label  # For counting connected components
#chỗ này là chỗ hay này, mình import sys và os, sau đó mình clone git vào, vì thư mục mình tổ chức train và test thư mục khác nhau trong khi đó
#minh muốn test mô hình đã train, mô hình đã train thì phải import unetse và Unet ở train, nên mình dùng lệnh
#lệnh này sys.path.append(os.path.abspath('/content/segatten/train'))
sys.path.append(os.path.abspath('/content/segatten/train'))
from unetse import Unet

# Setup CUDA
def setup_cuda():
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict(in_file, img_size=480):
    """
    :param in_file: image file
    :param img_size: image size
    """
    model.eval()

    # Pre-process input image and its ground-truth
    img = Image.open(in_file).convert('RGB')  # convert to RGB image
    W, H = img.size
  
    img_resized = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)(img)
    img_tensor = T.ToTensor()(img_resized).to(device, dtype=torch.float).unsqueeze(dim=0)

    # Perform a forward pass
    logits = model(img_tensor)

    # Upsample the output to the original resolution for a better visualization
    logits = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(logits)

    # Measure the segmentation performance
    seg_map = logits.cpu().detach().numpy().argmax(axis=1)
    seg_map = seg_map.squeeze()  # 'squeeze' used to remove the first dimension of 1 (i.e., batch size)

    # Count the number of segments (connected components)
    labeled_seg_map, num_segments = label(seg_map)
    print(f"Number of segments: {num_segments}")

    # Overlay the segment count on the image
    overlaid = visualize(seg_map, np.array(img))
    overlaid = Image.fromarray(overlaid)

    draw = ImageDraw.Draw(overlaid)
    
    # Optionally, you can use a custom font. If so, make sure the font is available in Colab.
    # font = ImageFont.truetype("arial.ttf", 36)
    font = ImageFont.load_default()
    
    text = f"Segments: {num_segments}"
    text_size = draw.textsize(text, font=font)
    text_position = (W - text_size[0] - 10, H - text_size[1] - 10)  # Position the text at the bottom-right corner
    draw.text(text_position, text, fill=(255, 255, 255), font=font)  # White text

    overlaid.save(cmd_args.output + os.sep + os.path.basename(in_file))

    print('File: ' + os.path.basename(in_file) + ' done')


if __name__ == "__main__":

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Test a post-mortem iris segmentation model')
    args.add_argument('-i', '--input', default=None, type=str, help='Path to the input image')
    args.add_argument('-w', '--weights', default='/content/drive/My Drive/AI/udeep/unetse/unetse_epoch_71_iou_0.8435.pt', type=str,
                      help='Trained weights')
    args.add_argument('-o', '--output', default='outputs', type=str, help='Output folder')
    cmd_args = args.parse_args()

    device = setup_cuda()

    # 2. Create a segmentation model, then load the trained weights
    model = Unet(in_ch=3, out_ch=2).to(device)
    model.load_state_dict(torch.load(cmd_args.weights, device))
    print('The segmentation model has been loaded.')

    # 3. Perform segmentation and count segments
    predict(cmd_args.input)

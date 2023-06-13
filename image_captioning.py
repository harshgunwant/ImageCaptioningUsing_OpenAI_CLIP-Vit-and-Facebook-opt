import torch
from train_and_eval_combined import FlamingoModel, FlamingoProcessor, load_url
from PIL import Image

print('preparing model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlamingoModel.from_pretrained('./flamingo-coco_2/checkpoint-618')
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)

# load and process an example image
print('loading image and generating caption...')
image = load_url(
    'https://raw.githubusercontent.com/rmokady/CLIP_prefix_caption/main/Images/CONCEPTUAL_02.jpg')
caption = model.generate_captions(processor, images=[image], device=device)
print('generated caption:', caption)

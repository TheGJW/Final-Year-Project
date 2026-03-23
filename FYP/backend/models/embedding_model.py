import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

# loading the pretrained mobilenet model that is trained on imagenet
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
model.eval()

# define processing pipeline for input images
transform = transforms.Compose([
    # resize image to fit input for mobilenet
    transforms.Resize((224,224)),
    # convert image to tensor
    transforms.ToTensor(),
    # normalize tensor using mean and standard deviation
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
# function to generate embedding vector from an image
def get_embedding(image_path):
    # load image and ensure it is in RGB format
    image = Image.open(image_path).convert("RGB")
    # applying preprocessign and adding batch dimensions
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # extracting feature maps from convolutional layers
        features = model.features(tensor)
        # applying global average pooling to reduce hxw to a 1x1
        features = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
        # flattening the pooled features into a 1D vector
        embedding = features.flatten()
        # normalizing embedding for similarity comparison
        embedding = embedding / embedding.norm()
    # converting the tensor to array and return
    return embedding.numpy()
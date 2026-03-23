import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class SceneClassifier:

    def __init__(self):
        # creating a resnet18 model with 365 output classes
        self.model = models.resnet18(num_classes=365)
        # loading the predownloaded local checkpoint file
        # also ensure that it is ran on cpu
        checkpoint = torch.load(
            "weights/resnet18_places365.pth.tar",
            map_location="cpu"
        )

        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        # loading the processed weights into the model
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # processing pipeline for input images
        # they are adjusted to fit the requirements for model input
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # loading the scene category labels from file
        with open("weights/categories_places365.txt") as f:
            # extract the clean class names by removing the index prefix
            self.classes = [line.strip().split(' ')[0][3:] for line in f]

        # define common indoor scenes for filtering
        self.allowed_scenes = {
            "bedroom",
            "living_room",
            "family_room",
            "lounge",
            "kitchen",
            "dining_room",
            "bathroom",
            "closet",
            "laundry_room",
            "office",
            "home_office",
            "studio"
        }
    # function to classify scene from an image
    def classify(self, image_path, threshold=0.35):
        # loading image and converting to RGB format
        img = Image.open(image_path).convert("RGB")
        # applying preprocessing and adding batch dimension
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            # getting the raw model output
            output = self.model(img)
        # converting logits to probabilities using softmax
        probs = F.softmax(output, dim=1)
        # get highest probability and corresponding class index
        confidence, pred = torch.max(probs, 1)
        # convert tensor values to python scalars
        confidence = confidence.item()
        scene = self.classes[pred.item()]

        # filter 1: rejecting low confidence predictions
        if confidence < threshold:
            return "unknown_scene", confidence

        # filter 2: reject scenes not in allowed indoor list
        if scene not in self.allowed_scenes:
            return "unknown_scene", confidence
        # return the scene predicted and the confidence score
        return scene, confidence
    
import torchvision
pretrained_model = dict()

pretrained_model = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "densenet121": torchvision.models.densenet121,
    
}
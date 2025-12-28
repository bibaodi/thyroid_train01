from torchvision import transforms


class ImageKeepRatioResize:
    def __init__(self, target_width=224, target_height=224):
        self.target_width = target_width
        self.target_height = target_height
    
    def __call__(self, image):
        w, h = image.size
        
        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = transforms.Resize((new_h, new_w))(image)
        
        pad_w = (self.target_width - new_w) // 2
        pad_h = (self.target_height - new_h) // 2
        
        pad_transform = transforms.Pad(
            (pad_w, pad_h, self.target_width - new_w - pad_w, self.target_height - new_h - pad_h), 
            fill=0
        )
        return pad_transform(image)


def get_transforms_keep_aspect_ratio(input_image_size=(224, 224)):
    target_W, target_H = input_image_size

    resize_and_pad_tsfm = ImageKeepRatioResize(target_width=target_W, target_height=target_H)

    train_transform = transforms.Compose([
        resize_and_pad_tsfm,
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        resize_and_pad_tsfm,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class ImageCenterPadWithoutScaleUp:
    def __init__(self, target_width=224, target_height=224):
        self.target_width = target_width
        self.target_height = target_height
    
    def __call__(self, image):
        w, h = image.size
        
        if w > self.target_width or h > self.target_height:
            scale = min(self.target_width / w, self.target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = transforms.Resize((new_h, new_w))(image)
            
        padded_w, padded_h = image.size
        pad_w = (self.target_width - padded_w) // 2
        pad_h = (self.target_height - padded_h) // 2
        
        pad_transform = transforms.Pad(
            (pad_w, pad_h, self.target_width - padded_w - pad_w, self.target_height - padded_h - pad_h), 
            fill=0
        )
        return pad_transform(image)


def get_transforms_center_pad_only(input_image_size=(224, 224)):
    target_W, target_H = input_image_size
    
    center_pad_tsfm = ImageCenterPadWithoutScaleUp(target_width=target_W, target_height=target_H)
    
    train_transform = transforms.Compose([
        center_pad_tsfm,
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        center_pad_tsfm,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

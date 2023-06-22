

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img,mask


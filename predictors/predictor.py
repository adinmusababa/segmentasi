import numpy as np
from PIL import Image

from models.deeplab import DeepLab
from data_generators.data_generator import initialize_data_loader
from utils.metrics import Evaluator
from tqdm import tqdm
from losses.loss import SegmentationLosses
import torch
from preprocessing.custom_transforms import Normalize, ToTensor, FixedResize

class Predictor():
    def __init__(self, config, checkpoint_path='./experiments/checkpoint_best.pth.tar'):
        self.config = config
        self.checkpoint_path = checkpoint_path

        # Class names for plant dataset (20 classes: 0=background, 1-19=plant organs)
        self.categories_dict = {i: f"class_{i}" for i in range(config['network']['num_classes'])}
        self.categories_dict[0] = "background"
        self.categories_dict_rev = {v: k for k, v in self.categories_dict.items()}

        self.model = self.load_model()
        self.train_loader, self.val_loader, self.test_loader, self.nclass = initialize_data_loader(config)

        self.num_classes = self.config['network']['num_classes']
        self.evaluator = Evaluator(self.num_classes)
        self.criterion = SegmentationLosses(weight=None, cuda=self.config['network']['use_cuda']).build_loss(mode=self.config['training']['loss_type'])

        # Preprocessing transforms (same as validation)
        self.crop_size = config['image']['crop_size']
        self.means = (0.485, 0.456, 0.406)
        self.stds = (0.229, 0.224, 0.225)
        self.resize_transform = FixedResize(self.crop_size)
        self.normalize_transform = Normalize(mean=self.means, std=self.stds)
        self.to_tensor_transform = ToTensor()


    def load_model(self):
        model = DeepLab(num_classes=self.config['network']['num_classes'], backbone=self.config['network']['backbone'],
                        output_stride=self.config['image']['out_stride'], sync_bn=False, freeze_bn=True)


        if self.config['network']['use_cuda']:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location={'cuda:0': 'cpu'}, weights_only=False)

#        print(checkpoint)
        model = torch.nn.DataParallel(model)

        model.load_state_dict(checkpoint['state_dict'])

        return model

    def inference_on_test_set(self):
        print("inference on test set")

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.config['network']['use_cuda']:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print("Accuracy:{}, Accuracy per class:{}, mean IoU:{}, frequency weighted IoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)



    def segment_image(self, filename):
        """Segment a single image without ground truth mask.

        Args:
            filename: Path to input image

        Returns:
            image: Original image as numpy array (H, W, 3), uint8
            prediction: Predicted mask as numpy array (H, W), int64 (class indices)
        """
        img = Image.open(filename).convert('RGB')
        orig_w, orig_h = img.size

        # Preprocess (same as validation: FixedResize -> Normalize -> ToTensor)
        sample = {'image': img, 'label': img}
        sample = self.resize_transform(sample)
        sample = self.normalize_transform(sample)
        sample = self.to_tensor_transform(sample)
        image = sample['image'].unsqueeze(0)  # (1, 3, H, W)

        if self.config['network']['use_cuda']:
            image = image.cuda()

        with torch.no_grad():
            prediction = self.model(image)  # (1, num_classes, H, W)

        # Get prediction mask
        pred = prediction.squeeze(0).cpu().numpy()  # (num_classes, H, W)
        pred_mask = np.argmax(pred, axis=0).astype(np.uint8)  # (H, W)

        # Resize prediction back to original image size
        pred_mask_pil = Image.fromarray(pred_mask).resize((orig_w, orig_h), Image.NEAREST)
        pred_mask = np.array(pred_mask_pil)

        # Denormalize original image for visualization
        orig_img = np.array(img).astype(np.uint8)

        return orig_img, pred_mask
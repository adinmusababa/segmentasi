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

        self.num_classes = config['network']['num_classes']
        self.categories_dict = {i: f"class_{i}" for i in range(self.num_classes)}
        self.categories_dict[0] = "background"
        if self.num_classes == 2:
            self.categories_dict[1] = "leaf"
        self.categories_dict_rev = {v: k for k, v in self.categories_dict.items()}

        self.model = self.load_model()
        self.train_loader, self.val_loader, self.test_loader, self.nclass = initialize_data_loader(config)

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

        state_dict = checkpoint['state_dict']
        # Handle both DataParallel ('module.') and plain model checkpoints
        keys = list(state_dict.keys())
        wrapped = keys[0].startswith('module.')
        requires_prefix = not wrapped
        if requires_prefix:
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        return model


    def inference_on_test_set(self):
        """Evaluate on the actual test split (not val_loader).

        Reports foreground (leaf) metrics: IoU, Dice, Precision, Recall,
        plus pixel accuracy and confusion matrix.
        """
        print("inference on test set")

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
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

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        # Foreground (leaf) metrics from confusion matrix
        cm = self.evaluator.confusion_matrix
        tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou_leaf = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice_leaf = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        print("Accuracy:{}, Accuracy per class:{}, mean IoU:{}, frequency weighted IoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print('Foreground (leaf):')
        print("  IoU: {:.4f}, Dice: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(iou_leaf, dice_leaf, prec, rec))
        print('Confusion matrix (rows=target, cols=pred):')
        print(cm)

        return {
            'Acc': float(Acc), 'Acc_class': float(Acc_class),
            'mIoU': float(mIoU), 'fwIoU': float(FWIoU),
            'IoU_leaf': float(iou_leaf), 'Dice_leaf': float(dice_leaf),
            'Precision': float(prec), 'Recall': float(rec),
        }


    def predict_probability(self, filename):
        """Forward-probability of the foreground (leaf) class.

        Applies the same preprocessing pipeline as training, returns the
        foreground probability map at the original image spatial size.
        This is the output used for ensemble with U-Net: softmax channel 1.
        """
        img = Image.open(filename).convert('RGB')
        orig_w, orig_h = img.size

        sample = {'image': img, 'label': img}
        sample = self.resize_transform(sample)
        sample = self.normalize_transform(sample)
        sample = self.to_tensor_transform(sample)
        image = sample['image'].unsqueeze(0)  # (1, 3, H, W)

        if self.config['network']['use_cuda']:
            image = image.cuda()

        with torch.no_grad():
            logits = self.model(image)  # (1, num_classes, H, W)

        prob = torch.softmax(logits, dim=1)[:, 1]  # foreground channel
        prob = prob.squeeze(0).cpu().numpy()  # (H, W)

        # resize probability back to original image size
        prob_pil = Image.fromarray(prob.astype(np.float32)).resize((orig_w, orig_h), Image.BILINEAR)
        prob = np.array(prob_pil, dtype=np.float32)

        return prob


    def predict_mask(self, filename, threshold=0.5):
        """Binary mask from foreground probability (0/1)."""
        prob = self.predict_probability(filename)
        mask = (prob >= threshold).astype(np.uint8)
        return mask


    def segment_image(self, filename):
        """Segment a single image without ground truth mask.

        Args:
            filename: Path to input image

        Returns:
            image: Original image as numpy array (H, W, 3), uint8
            prediction: Predicted mask as numpy array (H, W), uint8 (0/1)
        """
        img = Image.open(filename).convert('RGB')
        orig_w, orig_h = img.size

        pred_mask = self.predict_mask(filename)

        # Original image for visualization
        orig_img = np.array(img).astype(np.uint8)

        return orig_img, pred_mask
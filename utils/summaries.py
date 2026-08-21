import os
import torch
from torchvision.utils import make_grid
from utils.datagen_utils import decode_seg_map_sequence

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


class _NullWriter(object):
    """No-op writer used when tensorboardX is not installed."""
    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        if SummaryWriter is None:
            print('Warning: tensorboardX not installed, tensorboard logging disabled.')
            return _NullWriter()
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, value_range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, value_range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
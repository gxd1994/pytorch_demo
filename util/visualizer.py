import os
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class Visualizer():
    def __init__(self, root):
        self.writer = SummaryWriter(log_dir=os.path.join(root, "vis_log"))


    def __getattr__(self, item):
        return getattr(self.writer, item)


    def add_images(self, name, img_tensor, global_step, normalize=True, scale_each=False):
        x = vutils.make_grid(img_tensor, normalize=normalize, scale_each=scale_each)
        self.writer.add_image(name, x, global_step=global_step)


    def plot_line(self, name, data, iter):
        self.writer.add_scalar(name, data, global_step=iter)

    def plot_lines(self, data, iter):
        for k, v in data.items():
            self.writer.add_scalar(k, v, global_step=iter)

    def plot_images(self, data, iter):
        for k, v in data.items():
            self.add_images(k, v, global_step=iter)

    def plot_images_prefix(self, data, iter, prefix="val"):
        for k, v in data.items():
            self.add_images(prefix + "/" + k, v, global_step=iter)


    def plot_lines_prefix(self, data, iter, prefix="val"):
        for k, v in data.items():
            self.writer.add_scalar(prefix + "/" + k, v, global_step=iter)
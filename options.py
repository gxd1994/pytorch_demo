import os
import sys
import shutil
import argparse
import util

class BaseOptions():
    def __init__(self):

        self.opt = None
        self.parser = argparse.ArgumentParser()

        # config for dataloader
        self.parser.add_argument('--name', type=str, default='config',
                                 help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        self.parser.add_argument('--load_scalar', type=int, default=1.2, help='scale images to this size')
        self.parser.add_argument('--fine_size', type=int, default=224, help='then crop to this size')

        # self.parser.add_argument('--num_channels', type=int, default=3,
        #                          help='image channels of inputs')
        # config for models

        self.parser.add_argument('--model', type=str, default='models.classifier.classifier',
                                 help='chooses which model to use.')

        self.parser.add_argument('--num_class', type=int, default=2,
                                 help='image channels of inputs')

        # config for experiments
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


    def parse(self):

        self.opt = self.parser.parse_args()

        # set gpu environ
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_ids

        args = vars(self.opt)
        args["logfile"] = os.path.join(self.opt.checkpoints_dir, "log.txt")

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # mkdir checkpints dir
        expr_dir = os.path.join(self.opt.checkpoints_dir)
        util.mkdirs(expr_dir)


        #save save_results_path
        util.mkdirs(self.opt.save_results_path)

        file_name = os.path.join(expr_dir, 'opt' + '.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        # save script to the disk
        file_name = os.path.join(expr_dir, 'run' + '.sh')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('%s\n' % ' '.join(sys.argv))

        return self.opt


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()


        #expr save path
        #self.parser.add_argument('--name', type=str, default='config',
        #                         help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints/bs32', help='models are saved here')
        self.parser.add_argument('--lr_policy', type=str, default='lambda')

        #train dataset path
        self.parser.add_argument('--dataroot_train', type=str, default='../data/hymenoptera_data/train.txt', help='dataset path prefix')
        #validation dataset_path
        self.parser.add_argument('--dataroot_val', type=str, default='../data/hymenoptera_data/val.txt', help='dataset path prefix')
        #dataset prefix
        self.parser.add_argument('--path_prefix', type=str, default='../data/hymenoptera_data', help='dataset path prefix')


        self.parser.add_argument('--train_collection', type=str, default='train_collection', help='train_collection')
        self.parser.add_argument('--val_collection', type=str, default='val_collection', help='val_collection')


        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')

        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')

        self.parser.add_argument('--save_epoch_freq', type=int, default=20,
                                 help='frequency of saving checkpoints at the end of epochs')

        # self.parser.add_argument('--eval_freq', type=int, default=500,
        #                          help='frequency of testing on evaluation dataset')

        # config for training mode
        self.parser.add_argument('--max_epoch', type=int, default=300, help='# of total epochs for training')

        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')

        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')

        self.parser.add_argument('--restore_spec_model', type=str, default=None, help='restrore from specific model')
        self.parser.add_argument('--save_results_path', type=str, default="./results", help='restrore from specific model')


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()

        self.parser.add_argument('--test_dir', type=str, default='../test_dir', help='test dir.')
        self.parser.add_argument('--restore_spec_model', type=str, default=None, help='restrore from specific model')
        self.parser.add_argument('--save_results_dir', type=str, default='../test_results_dir', help='save_results_dir')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/bs2', help='models are saved here')

        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

if __name__ == '__main__':
    import sys
    opt = TrainOptions()
    sys.argv.append('--dataroot=.\\data')
    print(sys.argv)
    opt.parse()

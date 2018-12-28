import torch
from dataloader import CreatDataLoader
from options import TrainOptions
import util
from util.visualizer import Visualizer
import time, os
from tqdm import tqdm


def print_current_states(log_name, epoch, i, states, t, phase="train"):
    message = '[phase:{:<15}, epoch:{:<5}, iters:{:<5}, time:{:<5.4f}] '.format(phase, epoch, i, t)
    for k, v in states.items():
        message += '%s: %.3f ' % (k, v)

    print(message)

    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)


def restore_model(model, opt):
    start_epoch = 1
    restore_path = opt.restore_spec_model
    print("restore_path", restore_path)

    if restore_path:
        if os.path.isfile(restore_path):
            checkpoint = torch.load(restore_path)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(restore_path, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
            model.model.load_state_dict(checkpoint['state_dict'])
            model.optim.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(restore_path))
    else:
        print("restore fail ")


    return start_epoch



def train(opt):
    dataloaders = {}
    dataloaders["train"] = CreatDataLoader(root=opt.dataroot_train, batch_size=opt.batch_size, fineSize=224, loadScalar=opt.load_scalar, degrees=15, phase="train", prefix_root=opt.path_prefix)
    dataloaders["val"] = CreatDataLoader(root=opt.dataroot_val, batch_size=opt.batch_size, fineSize=224, loadScalar=opt.load_scalar, degrees=None, phase="val", prefix_root=opt.path_prefix)

    model = util.parse_attr(opt.model)()
    model.initialize(opt)
    model.setup()
    start_epoch = restore_model(model, opt)
    model.update_lr_scheduler(start_epoch)

    vis = Visualizer(opt.checkpoints_dir)

    global_step = (start_epoch - 1) * len(dataloaders["train"].dataset) // opt.batch_size

    for epoch in range(start_epoch, opt.max_epoch):

        for phase in ["train", "val"]:
            running_loss, running_acc = 0.0, 0.0

            # for i, data in enumerate(tqdm(dataloaders[phase], desc="{phase}".format(phase=phase))):
            for i, data in enumerate(dataloaders[phase]):
                iter_start_time = time.time()
                model.set_input(data)

                if phase == "train":
                    global_step += 1
                    with torch.set_grad_enabled(True):
                        model.optimize_parameters()

                else:
                    model.eval()
                    with torch.set_grad_enabled(False):
                        model.forward()

                states = model.get_current_states()

                if phase == "train":

                    # if global_step % opt.print_freq == 0:
                    #     t = time.time() - iter_start_time
                    #     print_current_states(opt.logfile, epoch, i + 1, states.scalars, t)

                    if global_step % opt.display_freq == 0:
                        # visilization
                        vis.plot_lines(states.scalars, iter=global_step)
                        vis.plot_images(states.images,  iter=global_step)

                ''' 
                #check the input of validation        
                if phase == "val":
                    if global_step % opt.display_freq == 0:
                        # visilization
                        vis.plot_images_prefix(states.images, iter=global_step, prefix="val")
                '''

                running_loss += states.scalars["loss"] * data["img"].size(0)
                running_acc  += states.scalars["acc"] * data["img"].size(0)


            if epoch % opt.save_epoch_freq == 0:
                model.save_model(epoch, name="{}".format(global_step))
            if phase == "train":
                model.update_lr() #base epcoh

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)
            print_current_states(opt.logfile, epoch, i=0, states={"loss":epoch_loss, "acc":epoch_acc}, t=0, phase=phase+"_epoch")
            vis.plot_lines_prefix({"loss":epoch_loss, "acc":epoch_acc}, iter=global_step, prefix="val")


def main():
    opt = TrainOptions().parse()
    train(opt)



if __name__ == "__main__":
    main()

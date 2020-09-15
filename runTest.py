'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
from utils import load_datasets, make_dataloaders
from model import load_model, load_weights
import numpy as np
import torch
from train import Score_Observer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time
from utils import *
from localization import export_gradient_maps

def test(model, test_loader):
    print("Running test")
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    score_obs = Score_Observer('AUROC')
    # evaluate
    model.to(c.device)
    model.eval()
    epoch = 0
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    test_loss = list()
    test_z = list()
    test_labels = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            z = model(inputs)
            loss = get_loss(z, model.nf.jacobian(run_forward=False))
            test_z.append(z)
            test_loss.append(t2np(loss))
            test_labels.append(t2np(labels))

    test_loss = np.mean(np.array(test_loss))
    if c.verbose:
         print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
    score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                    print_score=c.verbose or epoch == c.meta_epochs - 1)

    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

def load_testloader(data_dir_test):
    def target_transform(target):
        return class_perm[target]

    classes = os.listdir(data_dir_test)
    if 'good' not in classes:
        print('There should exist a subdirectory "good". Read the doc of this function for further information.')
        exit()
    classes.sort()
    class_perm = list()
    class_idx = 1
    for cl in classes:
        if cl == 'good':
            class_perm.append(0)
        else:
            class_perm.append(class_idx)
            class_idx += 1

    augmentative_transforms = []
    if c.transf_rotations:
        augmentative_transforms += [transforms.RandomRotation(180)]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                                           saturation=c.transf_saturation)]

    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean, c.norm_std)]

    transform_train = transforms.Compose(tfs)
    testset = ImageFolderMultiTransform(data_dir_test, transform=transform_train, target_transform=target_transform,
                                        n_transforms=c.n_transforms_test)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size_test, shuffle=True,
                                             drop_last=False)
    return testloader

##########################  Main ####################
# train_set, test_set = load_datasets(c.dataset_path, c.class_name)
# _, test_loader = make_dataloaders(train_set, test_set)

test_loader = load_testloader("zerobox_dataset/zerobox_class/test")
model = torch.load("model_zerobox_test")

print("starting to run tests after loaded model and test dataset")
time_start = time.time()
# model = load_model(c.modelname)
test(model, test_loader)
time_end = time.time()
time_c = time_end - time_start  # 运行所花时间
print("time cost: {:f} s".format(time_c))

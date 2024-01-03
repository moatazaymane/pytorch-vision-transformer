from tqdm.notebook import tqdm
import os
import torch
from torchvision.datasets import CIFAR10
from utils.config import *
from torch.utils.data import DataLoader
import json
from dataset import VitDataset
from vit import vit_instance


def adaptive_lr(decay_epochs: int, decay: int, current_epoch: int,
                optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    if current_epoch % decay_epochs != 0:
        return optimizer

    for parameter in optimizer.param_groups:
        parameter['lr'] /= decay

    return optimizer


def val_accuracy(model, dl, device, iterator, epoch, batch_accuracies, batch_sizes):
    total_correct, val_size = 0, 0
    model.eval()

    with torch.no_grad():
        p = 0

        for batch in dl:
            p += 1

            inp, target = batch["flattened_patches"].to(device), batch["target"].to(device)
            size = inp.shape[0]
            target = target.to(dtype=torch.int64).to(device)
            outputs = model(inp)

            predicted = torch.softmax(outputs, dim=1).argmax(dim=1).detach().cpu().tolist()
            target = target.detach().cpu().tolist()
            assert len(predicted) == len(target)

            correct = 0

            for i in range(len(predicted)):

                if int(predicted[i]) == int(target[i]):
                    correct += 1

            total_correct += correct
            val_size += size

        iterator.write('\n')
        iterator.write(
            f"Training Accuracy after training epoch {epoch} : {sum(batch_accuracies) / sum(batch_sizes):.2f}")
        iterator.write(f"Validation Accuracy after training epoch {epoch} : {total_correct / val_size:.2f}")

        iterator.write('\n')
        return float(total_correct / val_size)

        # if epoch % decay_epochs==0:

        # iterator.write(f"New Learning rate {last_lr} --> {lr:.6f}")


def train_val_loop(vit_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_test = CIFAR10(os.getcwd(), download=True, train=False)
    vit_model.to(device)
    optimizer = torch.optim.Adam(vit_model.parameters(), lr=lr_gap)
    loss_function = torch.nn.CrossEntropyLoss()

    step, start = 0, 0
    if load_pretrained:
        state = torch.load(model_path_l16)
        vit_model.load_state_dict(state['model_state_dict'])
        start = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        step = state['step']

    test_ds = VitDataset(data_test.data, data_test.targets, patch_size, num_classes, preprocess=False, transform=False)
    val_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    data_train, targets = torch.load(train_data_path), torch.load(targets_path)

    assert data_train.shape[0] == 3 * 50000

    Loss, Accuracy = {}, {}

    for epoch in range(start, 100):

        train_ds = VitDataset(data_train, targets, patch_size, num_classes, preprocess=False, transform=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        vit_model.train()
        iterator = tqdm(train_dl, desc=f"epoch {epoch:02d}")
        batch_accuracies = []
        batch_sizes = []

        for batch in iterator:

            inp, target = batch["flattened_patches"].to(device), batch["target"].to(device)
            batch_len = inp.shape[0]
            target = target.to(dtype=torch.int64).to(device)
            output = vit_model.forward(inp)

            # Training Accuracy
            predicted_ = torch.softmax(output, dim=1).argmax(dim=1).detach().cpu().tolist()
            target_ = target.detach().cpu().tolist()
            assert len(predicted_) == len(target_)

            batch_correct = 0
            for i in range(len(predicted_)):

                if predicted_[i] == target_[i]:
                    batch_correct += 1

            batch_accuracies.append(batch_correct)
            batch_sizes.append(batch_len)

            loss = loss_function(output.view(-1, num_classes).to(device),
                                 target.type(torch.LongTensor).view(-1).to(device))
            iterator.set_postfix({'Loss': f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            accuracy = val_accuracy(vit_model, val_dl, device, iterator, epoch, batch_accuracies, batch_sizes)
            Loss[epoch] = float(loss.item())
            Accuracy[epoch] = float(accuracy)

            with open(loss_path_l16, "w") as f:
                json.dump(Loss, f)

            with open(accuracy_path_l16, "w") as f:
                json.dump(Accuracy, f)

            optimizer = adaptive_lr(10, 5, epoch, optimizer)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": vit_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            },
            model_path_l16
        )
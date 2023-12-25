from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from vit import vit_instance
from dataset import VitDataset
from utils.config import imgsize, patch_size, n_channels, D, L, k, Dmlp, num_classes, model_path, load_pretrained, \
    lr_cls, n_epochs, batch_size

vit_model = vit_instance(imgsize=imgsize, patch_size=patch_size, n_channels=n_channels, width=D, L=L, k=k, Dmlp=Dmlp,
                         num_classes=num_classes, dropout=0.1)

root = os.getcwd()
train_images = CIFAR10(root, download=True, train=True)
test_images = CIFAR10(root, download=True, train=False)
ds, test_ds = (VitDataset(train_images.data, train_images.targets, patch_size, len(train_images.targets)),
               VitDataset(test_images.data, test_images.targets, patch_size, len(train_images.targets)))

train_dl, val_dl = (DataLoader(ds, batch_size=batch_size, shuffle=True),
                    DataLoader(test_ds, batch_size=batch_size, shuffle=False))


def val_accuracy(model, dl, device, iterator):
    total_correct, val_size = 0, 0
    model.eval()

    with torch.no_grad():
        p = 0

        for batch in dl:
            p += 1

            inp, target = batch["flattened_patches"].to(device), batch["target"].to(device)
            size = inp.shape[0]
            target = target.to(dtype=torch.int64).to(device)
            outputs = vit_model(inp)

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
        iterator.write(f"Validation Accuracy after training epoch : {total_correct / val_size:.2f}")
        iterator.write('\n')


def train_val_loop(model, train_dl: DataLoader, val_dl: DataLoader):
    iterator = tqdm(train_dl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(vit_model.parameters(), lr=lr_cls)
    loss_function = torch.nn.CrossEntropyLoss()

    step, start = 0, 0
    if load_pretrained:
        state = torch.load(model_path)
        start = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        step = state['step']

    for epoch in range(start, n_epochs):

        vit_model.train()
        iterator = tqdm(train_dl, desc=f"epoch {epoch:02d}")


        for batch in iterator:
            # vit_model.train()
            inp, target = batch["flattened_patches"].to(device), batch["target"].to(device)
            target = target.to(dtype=torch.int64).to(device)
            output = vit_model.forward(inp)
            loss = loss_function(output.view(-1, num_classes).to(device),
                                 target.type(torch.LongTensor).view(-1).to(device))
            iterator.set_postfix({'Loss': f"{loss.item():6.3f}"})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            step += 1
        val_accuracy(vit_model, val_dl, device, iterator)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": vit_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            },
            model_path
        )


train_val_loop(vit_model, train_dl, val_dl)

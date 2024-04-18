import torch
from torch import nn
import wandb
from tqdm import tqdm
import torch

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
wandb.login()

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available() else "cpu"
)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class BaseClassifier(nn.Module):

    def train_step(self, batch, criterion):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return loss, acc

    def val_step(self, batch, criterion):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)
        loss = criterion(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def val_end(self, outputs):
        for x in outputs:
            val_losses = ["val_loss"]
            val_acc = x["val_acc"]
        val_loss = torch.stack(val_losses).mean()
        val_acc = torch.stack(val_acc).mean()
        return {"val_loss": val_loss.item(), "val_acc": val_acc.item()}

    def batch_end(self, epoch, batch, loss, acc, num_exmp):
        wandb.log(
            {"epoch": epoch, "batch": batch, "loss": loss, "accuracy": acc},
            step=num_exmp,
        )
        print(
            "Batch [{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}".format(
                str(batch).zfill(4),
                loss,
                acc,
            )
        )

    def val_end(self, outputs):
        losses = [x["val_loss"] for x in outputs]
        acces = [x["val_acc"] for x in outputs]
        loss = torch.stack(losses).mean()
        acc = torch.stack(acces).mean()
        return loss, acc


@torch.no_grad
def evaluate(model, epoch, criterion, train_loader, val_loader):
    model.eval()

    def get_vals(loader):
        output = []
        for batch in loader:
            output.append(model.val_step(batch, criterion))
        return model.val_end(output)

    train_loss, train_acc = get_vals(train_loader)
    val_loss, val_acc = get_vals(val_loader)

    wandb.log(
        {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }
    )

    print(
        "Epoch {}, Train Accuracy: {:.4f}, Train Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation Loss: {:.4f}".format(
            str(epoch).zfill(2), train_acc, train_loss, val_acc, val_loss
        )
    )

    return


def model_init_train(model, hyper_params, train_loader, val_loader, run_name):
    with wandb.init(project="wildlife_fin_run", config=hyper_params):

        config = wandb.config
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        wandb.watch(model, criterion, log="all", log_freq=10)

        sample_no = 0
        for epoch in tqdm(range(config.epochs)):
            print()
            batch_no = 0
            model.train()
            for batch in train_loader:
                loss, acc = model.train_step(batch, criterion)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_no += 1
                sample_no += config["batch_size"]
                if batch_no % 10 == 0:
                    model.batch_end(epoch, batch_no, float(loss), acc, sample_no)

            evaluate(model, epoch + 1, criterion, train_loader, val_loader)

        dims = torch.zeros(1, 3, config["resize"], config["resize"]).to(device)
        torch.onnx.export(
            model,
            dims,
            run_name,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        wandb.save(run_name)

    return model

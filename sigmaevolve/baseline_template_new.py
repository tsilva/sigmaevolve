import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_experiment(device, train_ds, val_ds):
# EVOLVE-BLOCK-START
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    def loss_fn(batch):
        x, y = (t.to(device) for t in batch)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, logits, y
# EVOLVE-BLOCK-END

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def prepare_datasets():
    ds = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    train_ds, val_ds = random_split(ds, [55_000, 5_000], generator=torch.Generator().manual_seed(42))
    return train_ds, val_ds


def run_epoch(model, loader, loss_batch, optimizer=None):
    # Set model in training mode if optimizer is provided
    # (otherwise it will be in evaluation mode)
    is_training = optimizer is not None
    model.train(is_training)
    context = torch.enable_grad() if is_training else torch.no_grad()

    # Start epoch
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    with context:
        # Iterate over batches
        for batch in loader:
            # Compute loss, logits and targets
            loss, logits, y = loss_batch(batch)

            # If optimizer is provided, 
            # backprop loss and step optimizer
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update total loss, correct and number of samples
            n = y.size(0)
            total_loss += loss.item() * n
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += n

    # Return average loss and accuracy
    avg_loss = total_loss / total_n
    avg_accuracy = total_correct / total_n
    return avg_loss, avg_accuracy


def fit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = prepare_datasets()

    # Create experiment configuration
    exp = make_experiment(device, train_ds, val_ds)
    model = exp["model"]
    scheduler = exp["scheduler"]
    optimizer = exp["optimizer"]
    train_loader = exp["train_loader"]
    val_loader = exp["val_loader"]
    loss_fn = exp["loss_fn"]

    # Start training loop
    n_epochs = 5
    patience = 3
    min_delta = 0.0
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    early_stop = False
    for epoch in range(1, n_epochs + 1):
        # If early stop is True, break the loop
        if early_stop: break

        # Run training epoch
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)

        # Run validation epoch
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        
        # If scheduler is available take a step
        if scheduler is not None: scheduler.step(val_loss)

        # Print training and validation metrics
        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # In case validation loss improved store the best state and reset the bad epochs counter
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        # Otherwise increment the bad epochs counter and check if we should early stop
        else:
            bad_epochs += 1
            if bad_epochs >= patience: early_stop = True    

    # If we found a best state, load it into the model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"best_val_loss={best_loss:.4f}")


if __name__ == "__main__":
    fit()
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SiameseNetwork
from dataset import FeatureMatchingDataset
from loss import ContrastiveLoss
from utils import threshold_sigmoid, threshold_contrastive_loss, visualize_predictions

# ── Hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 10
NUM_EPOCHS = 5


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    """Train the Siamese network."""
    import torchvision.transforms as transforms

    default_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & loader
    train_dataset = FeatureMatchingDataset(
        args.data_dir, args.train_file, split="train", transform=default_transform
    )
    print(f"Loaded {len(train_dataset)} training pairs.")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    siamese_net = SiameseNetwork(args.contra_loss).to(device)

    # Loss
    if args.contra_loss:
        criterion = ContrastiveLoss(margin=args.margin)
        print(f"Using Contrastive Loss (margin={args.margin})")
    else:
        criterion = torch.nn.BCELoss()
        print("Using Binary Cross-Entropy Loss")

    # Optimizer
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=args.lr)

    # ── W&B (optional — skip gracefully if not installed / not logged in) ──
    use_wandb = False
    try:
        import wandb
        wandb.init(
            project="siamese-oxford5k",
            config={
                "epochs": args.epochs,
                "batch_size": BATCH_SIZE,
                "lr": args.lr,
                "loss": "contrastive" if args.contra_loss else "bce",
                "margin": args.margin,
            },
        )
        use_wandb = True
        print("W&B logging enabled.")
    except Exception as e:
        print(f"W&B not available ({e}). Continuing without it.")

    # ── Training loop ─────────────────────────────────────────────────
    train_losses = []
    num_epochs = args.epochs

    for epoch in range(1, num_epochs + 1):
        siamese_net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for img1, img2, labels in loop:
            # ── Move to device ────────────────────────────────────────
            # img1 / img2 may come back as numpy arrays from the dataset __getitem__
            # depending on whether ToTensor was applied before np.array conversion.
            # Convert safely.
            if not isinstance(img1, torch.Tensor):
                img1 = torch.tensor(img1).float()
            if not isinstance(img2, torch.Tensor):
                img2 = torch.tensor(img2).float()

            # If shape is (B, H, W, C) permute to (B, C, H, W)
            if img1.ndim == 4 and img1.shape[-1] in (1, 3):
                img1 = img1.permute(0, 3, 1, 2)
                img2 = img2.permute(0, 3, 1, 2)

            img1   = img1.to(device)
            img2   = img2.to(device)
            labels = labels.view(-1, 1).float().to(device)

            # ── Forward ───────────────────────────────────────────────
            optimizer.zero_grad()

            if args.contra_loss:
                out1, out2 = siamese_net(img1, img2)
                loss = criterion(out1, out2, labels.squeeze(1))
                # For accuracy: use distance threshold
                preds = threshold_contrastive_loss(out1, out2, args.margin)
            else:
                output = siamese_net(img1, img2)
                loss = criterion(output, labels)
                preds = threshold_sigmoid(output)

            # ── Backward & update ─────────────────────────────────────
            loss.backward()
            optimizer.step()

            # ── Metrics ───────────────────────────────────────────────
            running_loss += loss.item()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc  = 100.0 * correct / total if total > 0 else 0.0
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")

        if use_wandb:
            import wandb
            wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "epoch": epoch})

        # ── Periodic evaluation ───────────────────────────────────────
        if epoch % args.eval_freq == 0:
            eval_loader = torch.utils.data.DataLoader(
                FeatureMatchingDataset(
                    args.data_dir, args.train_file, split="test",
                    transform=default_transform
                ),
                batch_size=BATCH_SIZE, shuffle=False, num_workers=0
            )
            val_acc = evaluate(args, "validation", eval_loader, siamese_net)
            if use_wandb:
                import wandb
                wandb.log({"val_acc": val_acc, "epoch": epoch})

    # ── Post-training ─────────────────────────────────────────────────
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    print("Saved training curve → training_loss.png")

    # Save model
    torch.save(siamese_net.state_dict(), args.model_file)
    print(f"Saved model → {args.model_file}")

    if use_wandb:
        import wandb
        wandb.finish()

    return siamese_net


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(args, split, data_loader, siamese_net, visualize=False):
    """Evaluate the Siamese network on a given data split."""
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    siamese_net.eval()

    correct = 0.0
    total   = 0.0
    sample_imgs1, sample_imgs2, all_labels, all_preds = [], [], [], []

    with torch.no_grad():
        for img1_set, img2_set, labels in data_loader:
            if not isinstance(img1_set, torch.Tensor):
                img1_set = torch.tensor(img1_set).float()
            if not isinstance(img2_set, torch.Tensor):
                img2_set = torch.tensor(img2_set).float()
            if img1_set.ndim == 4 and img1_set.shape[-1] in (1, 3):
                img1_set = img1_set.permute(0, 3, 1, 2)
                img2_set = img2_set.permute(0, 3, 1, 2)

            labels = labels.view(-1, 1).float()

            img1_set = img1_set.to(device)
            img2_set = img2_set.to(device)
            labels   = labels.to(device)

            if args.contra_loss:
                output1, output2 = siamese_net(img1_set, img2_set)
                output_labels = threshold_contrastive_loss(output1, output2, args.margin)
            else:
                output_prob = siamese_net(img1_set, img2_set)
                output_labels = threshold_sigmoid(output_prob)

            total   += labels.size(0)
            correct += (output_labels == labels).sum().item()

            if visualize and len(sample_imgs1) < 5:
                for i in range(min(5 - len(sample_imgs1), labels.size(0))):
                    sample_imgs1.append(img1_set[i])
                    sample_imgs2.append(img2_set[i])
                    all_labels.append(labels[i])
                    all_preds.append(output_labels[i])

    accuracy = 100.0 * correct / total
    print(f"Accuracy on {int(total)} {split} images: {accuracy:.2f}%")

    if visualize and sample_imgs1:
        visualize_predictions(
            torch.stack(sample_imgs1),
            torch.stack(sample_imgs2),
            torch.stack(all_labels),
            torch.stack(all_preds)
        )

    siamese_net.train()
    return accuracy


# ── Test ──────────────────────────────────────────────────────────────────────
def test(args, siamese_net=None):
    """Test the Siamese network on the test set."""
    import torchvision.transforms as transforms

    if siamese_net is None:
        siamese_net = SiameseNetwork(args.contra_loss)
        siamese_net.load_state_dict(torch.load(args.model_file, map_location='cpu'))
        print(f"Loaded model from {args.model_file}")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    siamese_net = siamese_net.to(device)

    default_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = FeatureMatchingDataset(
        args.data_dir, args.train_file, split="test", transform=default_transform
    )
    print(f"Loaded {len(test_dataset)} testing pairs.")

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_acc = evaluate(args, "testing", test_loader, siamese_net, visualize=True)
    return test_acc


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Siamese Network for Feature Matching')
    parser.add_argument('--action', type=str,
                        choices=['train', 'test', 'train_test'],
                        default='train_test')
    parser.add_argument('--data_dir',   type=str, default='./')
    parser.add_argument('--train_file', type=str, default='./groundtruth.json')
    parser.add_argument('--model_file', type=str, default='siamese_model.pth')
    parser.add_argument('--epochs',     type=int, default=NUM_EPOCHS)
    parser.add_argument('--margin',     type=float, default=1.0)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--cuda',       action='store_true', default=False)
    parser.add_argument('--contra_loss',action='store_true', default=False)
    parser.add_argument('--eval_freq',  type=int, default=1)
    args = parser.parse_args()

    print(f"Arguments: {args}")

    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        test(args)
    elif args.action == 'train_test':
        siamese_net = train(args)
        test(args, siamese_net)


if __name__ == '__main__':
    main()
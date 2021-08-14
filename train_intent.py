import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange
from tqdm import tqdm

from dataset import SeqClsDataset
from utils import Vocab

from torch.utils.data import DataLoader
from model import SeqClassifier
from torch.optim import Adam
import torch.nn.functional as F

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    dataloader_train = DataLoader(datasets['train'], shuffle=True, batch_size=args.batch_size,
                                  collate_fn=datasets['train'].collate_fn)  # datasets['train'] : SeqClsDataset
    dataloader_eval = DataLoader(datasets['eval'], shuffle=True, batch_size=args.batch_size,
                                 collate_fn=datasets['eval'].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                          dropout=args.dropout, bidirectional=args.bidirectional,
                          num_class=datasets['train'].num_classes).to(args.device)
    # TODO: init optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        for idx, (batch, target) in enumerate(tqdm(dataloader_train)):
            if len(target) == args.batch_size:
                optimizer.zero_grad()
                output = model(batch)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        wrong_list = []
        acc_list = []
        with torch.no_grad():
            for idx, (batch, target) in enumerate(tqdm(dataloader_eval)):
                prediction = model(batch)
                ls = prediction.tolist()
                for i in range(len(target)):
                    if ls[i].index(max(ls[i])) == target.tolist()[i]:
                        acc_list.append(1)
                    else:
                        wrong_list.append(1)
        acc = len(acc_list) / (len(acc_list) + len(wrong_list))
        print(acc)
        torch.save(model.state_dict(), f'./ckpt/intent/model_{acc}.pkl')


# TODO: Inference on test set

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=20)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

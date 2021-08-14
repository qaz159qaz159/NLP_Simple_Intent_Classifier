import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import torch

from dataset_test import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                            collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                          dropout=args.dropout, bidirectional=args.bidirectional,
                          num_class=dataset.num_classes).to(device)
    model.load_state_dict(torch.load("./ckpt/intent/model_0.9150815217391305.pkl"))

    # TODO: predict dataset
    with open(args.pred_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                if len(batch) == args.batch_size:
                    prediction = model(batch)
                    ls = prediction.tolist()[0]
                    writer.writerow([data[idx]['text'], dataset.idx2label(ls.index(max(ls)))])

    # TODO: write prediction to file (args.pred_file)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        # required=True,
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        # required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=20)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

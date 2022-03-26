import argparse
from config import Config
from train import Trainer
import genDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This is a code for training model.''')
    parser.add_argument('--imageRoot', type=str, default=r'D:\2022\3\medicalSeg\data\images\images', help='path to the root of image')
    parser.add_argument('--jsonRoot', type=str,
                        default=r'D:\2022\3\medicalSeg\data\images\jsons',
                        help='path to the root of data')
    parser.add_argument('--in_chs', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='the number of class')
    args = parser.parse_args()

    print('The training dataset is preparing... Please wait!')
    genDataset.DataGeneration(args.imageRoot, args.jsonRoot)
    config = Config()
    config.in_chs = args.in_chs
    config.num_classes = args.num_classes

    print("Everything is ok! It's time for training.")
    trainer = Trainer(config)
    trainer.train()
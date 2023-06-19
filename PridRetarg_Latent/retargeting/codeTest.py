import sys
sys.path.append('./retargeting/')
from torch.utils.data.dataloader import DataLoader
from models import create_model
from datasets import create_dataset, get_character_names
import option_parser
import os
from option_parser import try_mkdir
import time

if __name__== '__main__':
    args = option_parser.get_args()
    characters = get_character_names(args)

    dataset = create_dataset(args, characters)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(characters)
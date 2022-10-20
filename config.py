import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', nargs="+", default=[0], type=int)

    parser.add_argument('--bn', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)

    parser.add_argument('--euler', action='store_true', default=False)
    parser.add_argument('--rgb', action='store_true', default=False)

    parser.add_argument('--save_name', default="saved_model.pt", type=str)
    parser.add_argument('--load_name', default="saved_model.pt", type=str)

    arg = parser.parse_args()

    return arg
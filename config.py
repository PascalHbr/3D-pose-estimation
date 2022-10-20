import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', nargs="+", default=[0], type=int)

    parser.add_argument('--bn', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--euler', action='store_true', default=False)
    parser.add_argument('--binary', action='store_true', default=False)
    parser.add_argument('--loss', default="mean_angle", type=str)

    arg = parser.parse_args()

    return arg
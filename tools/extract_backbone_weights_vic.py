import torch
import argparse

### need optimizers.py in the save directory

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    output_dict = dict(state_dict=dict(), author="OpenSelfSup")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    if "model" in ck:
        state_dict = ck["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
        output_dict['state_dict'] = state_dict
    torch.save(output_dict, args.output)
    print("Done converting weights.")


if __name__ == '__main__':
    main()

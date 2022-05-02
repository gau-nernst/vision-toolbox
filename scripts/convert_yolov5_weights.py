import argparse

import torch


def convert_weights(src_path, dst_path):
    weights = torch.load(src_path, map_location='cpu')
    new_state_dict = {}

    # RULES:
    # stem -> model.0
    # stages.0.conv -> model.1
    # stages.0.conv1 -> model.2.cv2
    # stages.0.conv2 -> model.2.cv1
    # stages.0.blocks.{i}.conv{j} -> model.2.m.{i}.cv{j}
    # stages.0.out_conv -> model.2.cv3

    for k, v in weights.items():
        if k.startswith('stem'):
            new_k = k.replace('stem', 'model.0')
        
        elif k.startswith('stages'):
            parts = k.split('.')
            stage_idx = int(parts[1])
            submodule_name = parts[2]
            
            if submodule_name == 'conv':
                new_k = '.'.join(['model', str(stage_idx * 2 + 1)] + parts[3:])
            
            elif submodule_name == 'conv1':
                new_k = '.'.join(['model', str(stage_idx * 2 + 2), 'cv2'] + parts[3:])

            elif submodule_name == 'conv2':
                new_k = '.'.join(['model', str(stage_idx * 2 + 2), 'cv1'] + parts[3:])

            elif submodule_name == 'blocks':
                parts[4] = parts[4].replace('conv', 'cv')
                new_k = '.'.join(['model', str(stage_idx * 2 + 2), 'm'] + parts[3:])

            elif submodule_name == 'out_conv':
                new_k = '.'.join(['model', str(stage_idx * 2 + 2), 'cv3'] + parts[3:])

            else:
                raise ValueError(f'Unexpected weight name: {k}')

        else:
            raise ValueError(f'Unexpected weight name: {k}')

        new_state_dict[new_k] = v
        print(f'{k} -> {new_k}. Shape: {tuple(v.shape)}')

    torch.save(new_state_dict, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path')
    parser.add_argument('dst_path')
    args = parser.parse_args()

    convert_weights(args.src_path, args.dst_path)


if __name__ == '__main__':
    main()

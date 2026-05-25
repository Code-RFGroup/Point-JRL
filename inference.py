import os
import torch
import numpy as np
import time
from tqdm import tqdm

from utils import utils


def main(config):
    model_name = config['global']['model_name']
    checkpoint_path = config['inference']['checkpoint']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config['global']['gpu_ids']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_parallel = torch.cuda.device_count() > 1

    print(f"=> Model: {model_name}")
    print(f"=> Loading checkpoint: {checkpoint_path}")

    model = utils.build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    if model_name == 'simsiam':
        encoder = model.encoder
        if hasattr(encoder, 'fc'):
            encoder.fc = torch.nn.Identity()
    elif model_name in ['pointnext', 'hybrid_point', 'hybrid_image']:
        encoder = model.encoder
    elif model_name == 'lpa':
        encoder = model
    elif model_name == 'gnn':
        encoder = model

    encoder.eval()
    if is_parallel:
        encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)

    train_loader, val_loader, test_loader = utils.build_dataloader(config)
    print(f"=> Start inference on {len(test_loader.dataset)} samples...")
    features_list = []
    filenames_list = []

    start_time = time.time()

    with torch.no_grad():
        for data, _, filenames in tqdm(test_loader):
            data = data.to(device)
            output = encoder(data)
            if model_name == 'gnn':
                feature = output[1]
            else:
                feature = output

            feature = torch.nn.functional.normalize(feature, dim=1)
            features_list.append(feature.cpu().numpy())
            filenames_list.extend(filenames)

    all_features = np.concatenate(features_list, axis=0)
    save_dir = utils.increment_dir(config['inference']['save_dir'])
    npy_path = os.path.join(save_dir, 'features.npy')
    txt_path = os.path.join(save_dir, 'filenames.txt')
    np.save(npy_path, all_features)
    with open(txt_path, 'w') as f:
        for name in filenames_list:
            f.write(name + '\n')

    print(f"=> Done! Time: {time.time() - start_time:.4f}s")
    print(f"=> Saved to {npy_path} (Shape: {all_features.shape})")


if __name__ == "__main__":
    config_path = './config.yaml'
    config = utils.load_config(config_path)
    utils.set_seed(config['global']['seed'])

    main(config)

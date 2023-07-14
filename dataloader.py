from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import Resize
from os.path import join
import os, torch, model
from tqdm import tqdm
import random

patch_size = 32
# pap = model.PatchAvgPooling(patch_size)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class DSet(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.data_path = random.sample(os.listdir(base_path), 10000)
        self.transform = Resize((64, 64))
        # for img_name in tqdm(os.listdir(base_path)):
        #     img = read_image(os.path.join(self.base_path, img_name))
        #     img = transform(img)
        #     L.append(img)
        # self.data = torch.stack(L) / 255
        
    def __len__(self):
        return self.data_path.__len__()

    def __getitem__(self, index):
        path = join(self.base_path, self.data_path[index])
        return self.transform(read_image(path)) / 255

T = 2000
ALPHA = 1-torch.linspace(1e-4, 2e-2, T)
def alpha(t):
    at = torch.prod(ALPHA[:t]).reshape((1, ))
    return torch.sqrt(torch.cat((at, 1-at)))
ALPHA_bar = torch.stack([alpha(t) for t in range(T)])#.to(device)

def left_collate_fn(X):
    # X = X.to(device)
    X = torch.stack(X)
    N = X.shape[0]
    t = torch.randint(0, T-1, (N, ))#, device=device)
    eps = torch.normal(0, 1, size=X.shape)#, device=device)
    a = torch.index_select(ALPHA_bar, 0, t)

    X_noise =  (
        a[:, 0].reshape((-1, 1, 1, 1)) * X + \
        a[:, 1].reshape((-1, 1, 1, 1)) * eps
    )#.float().to(device)
    # t = t.float().to(device)
    # t = t[None, None, ...].tile((X_noise.shape[0], X_noise.shape[1])).flatten(end_dim=2)
    eps = torch.cat(eps.chunk(4, dim=-1)[:3], dim=-1) #  W Chunk
    X_noise = torch.cat(X_noise.chunk(4, dim=-1)[:3], dim=-1)
    # eps = eps.flatten(end_dim=2)
    return X_noise, eps, t

def right_collate_fn(X):
    # X = X.to(device)
    X = torch.stack(X)
    N = X.shape[0]
    t = torch.randint(0, T-1, (N, ))#, device=device)
    eps = torch.normal(0, 1, size=X.shape)#, device=device)
    a = torch.index_select(ALPHA_bar, 0, t)

    X_noise =  (
        a[:, 0].reshape((-1, 1, 1, 1)) * X + \
        a[:, 1].reshape((-1, 1, 1, 1)) * eps
    )#.float().to(device)
    # t = t.float().to(device)
    # t = t[None, None, ...].tile((X_noise.shape[0], X_noise.shape[1])).flatten(end_dim=2)
    eps = torch.cat(eps.chunk(4, dim=-1)[1:], dim=-1) #  W Chunk
    X_noise = torch.cat(X_noise.chunk(4, dim=-1)[1:], dim=-1)
    # eps = eps.flatten(end_dim=2)
    return X_noise, eps, t

def get_train_loader(lengths, batch_size, workers = 8, shuffle=True, collate_fn=None):
    ds = DSet(r'M:\Yu-Che_Chang\ccbda\dfiltering_data_10000')
    # train_set, val_set = random_split(ds, lengths)
    # MultiEpochs
    train_loader = MultiEpochsDataLoader(
        ds, batch_size, 
        shuffle=shuffle, 
        num_workers=workers, 
        pin_memory=True, 
        # worker_init_fn=seed_worker,
        # generator=g
        collate_fn=collate_fn
    )
    # val_loader = DataLoader(
    #     val_set,
    #     batch_size, 
    #     shuffle=True, 
    #     num_workers=workers,
    #     pin_memory=True,
    #     # worker_init_fn=seed_worker,
    #     # generator=g
    # )
    return train_loader #, val_loader

# def get_test_loader(batch_size, workers = 8):
#     ts = TestSet('test')

#     test_loader = MultiEpochsDataLoader(
#         ts, batch_size,
#         shuffle=False,
#         num_workers=workers, 
#         pin_memory=True,
#     )
#     return test_loader
    

if __name__ == '__main__':
    base_path  = r'E:\303Kface\data'
    ds = DSet(base_path)
    print(len(ds))
    print(ds[0].shape)
    save_image(ds[0], 'ds0.png')

    print(get_train_loader([100, 7294-100], batch_size=10))
    
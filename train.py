import torch, time, os, datetime
from tqdm import tqdm
import model
#from model import PatchAvgPooling
from dataloader import get_train_loader, left_collate_fn, right_collate_fn# , get_test_loader

# from evaluate import evaluate
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import partial
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = 2000
ALPHA = 1-torch.linspace(1e-4, 2e-2, T)
def alpha(t):
    at = torch.prod(ALPHA[:t]).reshape((1, ))
    return torch.sqrt(torch.cat((at, 1-at)))
ALPHA_bar = torch.stack([alpha(t) for t in range(T)])#.to(device)

def train(model, data_loader, model_name, **config):
    start_time = time.time()

    epochs, loss_func, lr, optimizer, scheduler = config.values()
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer)
    
    pn = 64//32
    # pap = PatchAvgPooling(patch_size)

    lossl = []
    accl =[]
    # max_acc = 0.
    for epoch in range(1, epochs+1):
        train_loss = 0.
        model.train()
        for X_noise, eps, t in (bar := tqdm(data_loader)):

            # X = X.to(device)
            # N = X.shape[0]
            # t = torch.randint(0, T-1, (N, ))#, device=device)
            # eps = torch.normal(0, 1, size=X.shape)#, device=device)
            # a = torch.index_select(ALPHA_bar, 0, t)

            # X_noise =  pap(
            #     a[:, 0].reshape((-1, 1, 1, 1)) * X + \
            #     a[:, 1].reshape((-1, 1, 1, 1)) * eps
            # )#.float().to(device)
            # # t = t.float().to(device)
            # t = t[None, None, ...].tile((X_noise.shape[0], X_noise.shape[1])).flatten(end_dim=2).to(device)
            # eps = torch.stack(eps.chunk(X_noise.shape[1], dim=-1))
            # eps = torch.stack(eps.chunk(X_noise.shape[0], dim=-2))
            # X_noise = X_noise.flatten(end_dim=2).to(device)
            # eps = eps.flatten(end_dim=2).to(device)
            X_noise, eps, t = X_noise.to(device), eps.to(device), t.to(device)
            
            pred = model(X_noise, t)

            loss = loss_func(eps, pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            bar.set_description(f'[epoch{epoch:3d}|Training] lr={optimizer.param_groups[0]["lr"]:.2e}')
            bar.set_postfix_str(f'total loss {train_loss:5.2f}')

        # acc = evaluate(Encoder, test_loader, transform)
        scheduler.step(train_loss)
        lossl.append(train_loss)
        # accl.append(acc*100)
        # if max_acc < acc:
        #     max_acc = acc
        #     best_model = deepcopy()
        # print('max acc:', max_acc)
        if epoch % 10 == 0:
            PATH = os.path.join(r'M:\Yu-Che_Chang\ccbda\save', f'{model_name}_epoch{epoch}.pt')
            save_model(epoch, model, optimizer, PATH)
    best_model=None
    print(f'Training used {datetime.timedelta(seconds=time.time()-start_time)}')
    return lossl, accl, best_model

def save_model(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print('Save model')

config = {
    'epochs': 100,
    'loss_func': torch.nn.MSELoss(),#reduction='sum'),
    'lr': 1e-4,
    'optimizer': torch.optim.Adam,
    'scheduler': partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        factor=0.2,
        patience=10,
        mode='min',
        threshold=5e-3,
        verbose=True,
        min_lr=5e-5,
        cooldown=5
    )
}

config2 = {
    'epochs': 100,
    'loss_func': torch.nn.MSELoss(),#reduction='sum'),
    'lr': 1e-4,
    'optimizer': torch.optim.Adam,
    'scheduler': partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        factor=0.2,
        patience=10,
        mode='min',
        threshold=5e-3,
        verbose=True,
        min_lr=5e-5,
        cooldown=5
    )
}

batch_size = 64
step_every_n = 2
workers = 0

if __name__ == '__main__':

    M1 = model.Unet().to(device)
    M2 = model.Unet().to(device)
    # state_dict = torch.load(os.path.join('checkpoints', 'epoch30.pt'))
    # state_dict = torch.load(os.path.join('save', 'saved_point.pt'))
    # M.load_state_dict(state_dict)

    train_loader1 = get_train_loader(
        [], # meanless
        batch_size,
        workers,
        collate_fn = left_collate_fn
    )

    train_loader2 = get_train_loader(
        [], # meanless
        batch_size,
        workers,
        collate_fn = right_collate_fn
    )
    
    # test_loader = get_test_loader(
    #     batch_size = 500,
    #     workers = 1
    # )

    lossl, accl, best_model = train(M1, train_loader1, 'model_1', **config)
    train(M2, train_loader2, 'model_2', **config2)
    # PATH = os.path.join('save', 'saved_point.pt')
    # torch.save(M.state_dict(), PATH)

    x_ax = range(len(lossl))
    # lines = plt.plot(x_ax, accl, x_ax, lossl)
    lines = plt.plot(x_ax, lossl)
    plt.legend(lines, ('train loss'),
        loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})
    plt.show()



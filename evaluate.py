import model
import torch, os
from dataloader import get_train_loader
from tqdm import tqdm
from torchvision.utils import save_image

torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = 2000
ALPHA = (1-torch.linspace(1e-4, 2e-2, T)).reshape((-1, 1)).to(device)

def generate(model, gen_N, chan=3, resolu=(28, 28), T=1000, X_T=None):
    model.eval()
    
    with torch.no_grad():
        X_T = X_T if X_T!=None else torch.normal(0, 1, size=(gen_N, chan, *resolu), device=device)

        for t in (bar := tqdm(list(range(T))[::-1])):
            bar.set_description(f'[Denoising] step: {t}')
            z = torch.normal(0, 1, size=(gen_N, chan, *resolu), device=device) if t != 0 else torch.zeros((gen_N, chan, *resolu), device=device)

            a_bar = torch.prod(ALPHA[:t+1]).reshape((-1, 1, 1, 1))
            sig = torch.sqrt(1-ALPHA[t]).reshape((-1, 1, 1, 1))

            pred = model(X_T, torch.Tensor([t+1]*gen_N).to(device))
            X_T = 1/torch.sqrt(ALPHA[t]) * (X_T - (1-ALPHA[t])/torch.sqrt(1-a_bar) * pred)\
                + sig * z

        return X_T

def generate_and_save(model, gen_N, chan=3, resolu=(28, 28), T=1000, step=8):
    model.eval()
    L = []
    with torch.no_grad():
        # pos = torch.zeros((gen_N, ))
        X_T = torch.normal(0, 1, size=(gen_N, chan, *resolu)).to(device)

        for t in (bar := tqdm(list(range(T))[::-1])):
            bar.set_description(f'[Denoising] step: {t}')
            z = torch.normal(0, 1, size=(gen_N, chan, *resolu)).to(device) if t != 0 else torch.zeros((gen_N, chan, *resolu), device=device)

            a_bar = torch.prod(ALPHA[:t+1]).reshape((-1, 1, 1, 1))
            sig = torch.sqrt(1-ALPHA[t]).reshape((-1, 1, 1, 1))

            pred = model(X_T, torch.Tensor([t+1]*gen_N).to(device))
            X_T = 1/torch.sqrt(ALPHA[t]) * (X_T - (1-ALPHA[t])/torch.sqrt(1-a_bar) * pred)\
                + sig * z

            if t % (T//step) == 0:
                L.append(X_T)
            # L.append(X_T)
        save_image(torch.cat(L), 'process.png')

def LR_split(img):
    img_chunks = img.chunk(4, dim=-1)
    return torch.cat(img_chunks[:3], dim=-1), torch.cat(img_chunks[1:], dim=-1)
    
def generate_and_save_v2(model_l, model_r, gen_N, chan=3, resolu=(28, 28), T=1000, step=8):
    model_l.eval()
    model_r.eval()
    L = []
    with torch.no_grad():
        # pos = torch.zeros((gen_N, ))
        X_T = torch.normal(0, 1, size=(gen_N, chan, *resolu)).to(device)
        XTL, XTR = LR_split(X_T)

        for t in (bar := tqdm(list(range(T))[::-1])):
            bar.set_description(f'[Denoising] step: {t}')
            z = torch.normal(0, 1, size=(gen_N, chan, *resolu)).to(device) if t != 0 else torch.zeros((gen_N, chan, *resolu), device=device)

            a_bar = torch.prod(ALPHA[:t+1]).reshape((-1, 1, 1, 1))
            sig = torch.sqrt(1-ALPHA[t]).reshape((-1, 1, 1, 1))
            
            XTL, XTR = LR_split(X_T)
            pred_l = model_l(XTL, torch.Tensor([t+1]*gen_N).to(device)).chunk(3, dim=-1)
            pred_r = model_r(XTR, torch.Tensor([t+1]*gen_N).to(device)).chunk(3, dim=-1)
            pred_lap = (torch.cat(pred_l[1:], dim=-1) + torch.cat(pred_r[:2], dim=-1))
            pred = torch.cat((pred_l[0], pred_lap/2, pred_r[-1]), dim=-1)

            X_T = 1/torch.sqrt(ALPHA[t]) * (X_T - (1-ALPHA[t])/torch.sqrt(1-a_bar) * pred)\
                + sig * z

            
            if t % (T//step) == 0:
                L.append(X_T)
            # L.append(X_T)
        save_image(torch.cat(L), 'process.png')

def gen_by_models():
    M = model.SimpleUnet().to(device)
    L = []
    gen_N= 8
    resolu = (64, 64)
    X_T = torch.normal(0, 1, size=(gen_N, 3, *resolu), device=device)
    fns = os.listdir('checkpoints')
    fns.sort(key=lambda s:int(s[5:-3]))
    for cp in fns:
        print(cp)
        state_dict = torch.load(os.path.join('checkpoints', cp))
        M.load_state_dict(state_dict)
        L.append(generate(M, gen_N, resolu=resolu, X_T=X_T))
    save_image(torch.cat(L), 'gen_by_models.png')
        

if __name__ == '__main__':
    # gen_by_models()
    M_l = model.SimpleUnet().to(device)
    M_r = model.SimpleUnet().to(device)
    # state_dict = torch.load(os.path.join('checkpoints', 'epoch30.pt'))
    save_dict_l = torch.load(os.path.join(r'E:\303Kface\save', 'model_1_epoch100.pt'))
    M_l.load_state_dict(save_dict_l['model_state_dict'])

    save_dict_r = torch.load(os.path.join(r'E:\303Kface\save', 'model_2_epoch60.pt'))
    M_r.load_state_dict(save_dict_r['model_state_dict'])
    # test_loader = get_test_loader(
    #     batch_size = 500,
    #     workers = 1
    # )

    # data_loader = get_train_loader(
    #     [],
    #     256,
    #     2,
    #     shuffle=False
    # )
    

    ## Diffusion process
    generate_and_save_v2(M_l, M_r, 8, resolu=(64, 64), step=10, T=2000)
    # generate_and_save(M_r, 8, resolu=(64, 48), step=10, T=2000)
    
    ## Generative images
    # i = 0
    # while i < 1:
    #     imgs = generate(M, gen_N=100, resolu=(64, 64), T=1000)
    #     for img in (bar := tqdm(imgs)):
    #         i += 1
    #         bar.set_description('[Saving]')
    #         bar.set_postfix_str(f'{i:05d}.png')
            
    #         path = os.path.join('generative_images', f'{i:05d}.png')
    #         save_image(img, path)
    #         if i >= 10000:
    #             break
    #     print()


    
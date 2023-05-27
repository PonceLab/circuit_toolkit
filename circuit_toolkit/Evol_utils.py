import torch
import numpy as np

def Evol_experiment_FC6(scorer, optimizer, G, steps=100, init_code=None,):
    if init_code is None:
        init_code = np.random.randn(1, 4096)
    new_codes = init_code
    scores_all = []
    generations = []
    codes_all = []
    best_imgs = []
    for i in range(steps):
        codes_all.append(new_codes.copy())
        latent_code = torch.from_numpy(np.array(new_codes)).float()
        imgs = G.visualize(latent_code.cuda()).cpu()
        scores = scorer.score_tsr(imgs)
        new_codes = optimizer.step_simple(scores, new_codes, )
        print("step %d score %.3f (%.3f) (norm %.2f )" % (
                i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
        scores_all.extend(list(scores))
        generations.extend([i] * len(scores))
        best_imgs.append(imgs[scores.argmax(),:,:,:])

    final_imgs = imgs
    codes_all = np.concatenate(tuple(codes_all), axis=0)
    scores_all = np.array(scores_all)
    generations = np.array(generations)
    return codes_all, scores_all, generations, best_imgs, final_imgs


#%%
def Evol_experiment_BigGAN(scorer, optimizer, G, steps=100, RND=None, label="", init_code=None, batchsize=20):
    init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
    # optim_cust = CholeskyCMAES(space_dimen=256, init_code=init_code, init_sigma=0.2)
    new_codes = init_code + np.random.randn(25, 256) * 0.06
    scores_all = []
    generations = []
    for i in tqdm.trange(steps, desc="CMA steps"):
        imgs = G.visualize_batch_np(new_codes, B=batchsize)
        latent_code = torch.from_numpy(np.array(new_codes)).float()
        scores = scorer.score_tsr(imgs)
        print("step %d dsim %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
            i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
            latent_code[:, :128].norm(dim=1).mean()))
        new_codes = optimizer.step_simple(scores, new_codes, )
        scores_all.extend(list(scores))
        generations.extend([i] * len(scores))

    scores_all = np.array(scores_all)
    generations = np.array(generations)
    mtg = ToPILImage()(make_grid(imgs, nrow=7))
    mtg.save(join(savedir, "lastgen%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
    np.savez(join(savedir, "scores%s_%05d.npz" % (methodlab, RND)), generations=generations, scores_all=scores_all,
             codes_fin=latent_code.cpu().numpy())
    visualize_trajectory(scores_all, generations, title_str=methodlab).savefig(
        join(savedir, "traj%s_%05d_score%.1f.jpg" % (methodlab, RND, scores.mean())))
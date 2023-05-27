# dissection_toolkit
Common utility functions for insilico experiments for visual neuroscience

## Organization
```
```

## Installation

```bash
git clone https://github.com/PonceLab/circuit_toolkit
cd circuit_toolkit
pip install -e .
# to install all dependencies
pip install -r requirements.txt
```
## Demo Usage
Try out this self-contained demo in colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cbCaOt3xFFnQhB2bbIkf-C_Buf1wFVpL?usp=sharing)


### Recording from individual units

```python
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.layer_hook_utils import get_module_names
scorer = TorchScorer("alexnet", imgpix=224)
# Print out all layers in a model
get_module_names(scorer.model, (3, 224,224), device="cuda");
h = scorer.set_unit("score", ".features.ReLU11", (10,6,6), ingraph=False)
```
### Map Receptive Field

Here is an example of computing and plot receptive field of a unit.
```python
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss, grad_population_RF_estimate, show_gradmap
scorer = TorchScorer("alexnet", imgpix=224)
unit = ("alexnet", ".features.ReLU11", 10, 6, 6)
print("Unit %s" % (unit,))
gradAmpmap = grad_RF_estimate(scorer.model, ".features.ReLU11", (10,6,6), 
                        input_size=(3, 224, 224), device="cuda", show=True, 
                        reps=100, batch=10)
Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=1E-8, relthresh=0.01, square=True)
print("Xlim %s Ylim %s\nimgsize %s corner %s" % (
Xlim, Ylim, (Xlim[1] - Xlim[0], Ylim[1] - Ylim[0]), (Xlim[0], Ylim[0])))
fitdict = fit_2dgauss(gradAmpmap, f"{unit[0]}-"+unit[1], outdir="", plot=True)
```

### Selectivity of a natural image dataset
Here is an example of computing the selectivity of a natural image dataset `imagenette2-160`. 
```python
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from circuit_toolkit.CNN_scorers import TorchScorer
scorer = TorchScorer("alexnet", imgpix=224)
h = scorer.set_unit("score", ".features.ReLU11", (10,6,6), ingraph=False)
dataset = ImageFolder("imagenette2-160/train", transform=Compose([CenterCrop(130), 
                                                                  Resize(224),
                                                                  ToTensor(), ]))# Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
score_vec = []
label_vec = []
for imgs, labels in dataloader:
    scores = scorer.score_tsr(imgs, B=128,)
    score_vec.append(scores)
    label_vec.append(labels)

score_vec = np.concatenate(score_vec, axis=0)
label_vec = torch.cat(label_vec, dim=0)
```

### Run Evolution
Here is the basic version of the Evolution using FC6 GAN and alexnet as scorer. 
```python
import torch
import numpy as np
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.Optimizers import CholeskyCMAES
from circuit_toolkit.GAN_utils import upconvGAN
from circuit_toolkit.layer_hook_utils import get_module_names

scorer = TorchScorer("alexnet", imgpix=224)
h = scorer.set_unit("score", ".features.ReLU11", (10,6,6), ingraph=False)
G = upconvGAN("fc6").cuda().eval()
new_codes = np.random.randn(1, 4096)
optimizer = CholeskyCMAES(space_dimen=4096, init_code=new_codes, init_sigma=3.0,)
steps = 100
for i in range(steps):
    latent_code = torch.from_numpy(np.array(new_codes)).float()
    imgs = G.visualize(latent_code.cuda()).cpu()
    scores = scorer.score_tsr(imgs)
    new_codes = optimizer.step_simple(scores, new_codes, )
    print("step %d score %.3f (%.3f) (norm %.2f )" % (
            i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
```

More high level function `Evol_experiment_FC6` is also available.
```python
import numpy as np
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.Optimizers import CholeskyCMAES
from circuit_toolkit.GAN_utils import upconvGAN
from circuit_toolkit.Evol_utils import Evol_experiment_FC6

scorer = TorchScorer("alexnet", imgpix=224)
h = scorer.set_unit("score", ".features.ReLU11", (10,6,6), ingraph=False)
G = upconvGAN("fc6").cuda().eval()
new_codes = np.random.randn(1, 4096)
optimizer = CholeskyCMAES(space_dimen=4096, init_code=new_codes, init_sigma=3.0,)

codes_all, scores_all, generations, best_imgs, \
 final_imgs = Evol_experiment_FC6(scorer, optimizer, G, steps=100, init_code=new_codes)
```

### Manifold exploration experiments

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from circuit_toolkit.GAN_utils import upconvGAN
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.plot_utils import to_imgrid
from circuit_toolkit.GAN_manifold_utils import generate_sphere_grid_coords 
scorer = TorchScorer("alexnet", imgpix=224)
h = scorer.set_unit("score", ".features.ReLU11", (10, 6, 6), ingraph=False)
G = upconvGAN("fc6").cuda().eval()

vec_cnt = 3 * torch.randn(1, 4096).float()
codes = generate_sphere_grid_coords(vec_cnt, vec2=None, vec3=None,
                    n_az=9, n_el=9, az_lim=(-np.pi/2, np.pi/2),
                          el_lim=(-np.pi/2, np.pi/2))
manif_imgs = G.visualize_batch(codes, B=40)
to_imgrid(manif_imgs,nrow=9)

manif_scores = scorer.score_tsr(manif_imgs)
plt.imshow(manif_scores.reshape(9,9))
plt.show()
```

### Invert images into GAN space and exploration 
```python
from torchvision.transforms import ToTensor
from circuit_toolkit.GAN_utils import upconvGAN
from circuit_toolkit.GAN_invert_utils import GAN_invert
from circuit_toolkit.plot_utils import to_imgrid
G = upconvGAN("fc6").cuda().eval()
img = ...
target_img = ToTensor()(img)
z_opts, img_opts = GAN_invert(G, target_img.cuda(), max_iter=int(5E3),
                              print_progress=False)
to_imgrid([target_img,*img_opts.cpu()])
```


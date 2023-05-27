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

### Manifold exploration experiments


### Invert images into GAN space and exploration 



# Ordinary generative models
from .vae import VAE
from .dcgan import DCGAN
from .improvedgan import ImprovedGAN
from .ebgan import EBGAN
from .began import BEGAN
from .ali import ALI
from .ali_other import *
from .aae import AAE, BinAAE, AAE2, DrAAE2
from .hacked import DropoutALI
from .ali_svhn import ALIforSVHN

# Conditional generative models
from .cvae import CVAE
from .cvaegan import CVAEGAN
from .cali import CALI
from .triplegan import TripleGAN
from .hacked import *

# Image-to-image genearative models
from .cyclegan import CycleGAN
from .unit import UNIT

models = {
    'vae': VAE,
    'dcgan': DCGAN,
    'improvedgan': ImprovedGAN,
    'ebgan': EBGAN,
    'began': BEGAN,
    'ali': ALI,
    'wider_ali': WiderALI,
    'deeper_ali': DeeperALI,
    'local_conn_ali': LocallyConnALI,
    'mobile_ali': MobileNetALI,
    'vdcgan': VeryDcgan,
    'drdcgan': DropoutDcgan,
    'drvae': DropoutVae,
    'drimprovedgan': DropoutImprovedGAN,
    'drali': DropoutALI,
    'aae': AAE,
    'binaae': BinAAE,
    'aae2': AAE2,
    'draae2': DrAAE2,
    'vdimprovedgan': VeryDeepImprovedGAN,
    'ali_SVHN': ALIforSVHN,
}
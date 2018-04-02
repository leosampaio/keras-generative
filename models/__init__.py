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
from .ali_svhn import *
from .ali_mnist import *
from .alice import ALICE
from .alice_svhn import ALICEforSVHN, ALICEwithDSforSVHN
from .alice_mnist import ALICEforMNIST, ALICEwithDSforMNIST, ALICEforSharedExp

# Conditional generative models
from .cvae import CVAE
from .cvaegan import CVAEGAN
from .cali import CALI
from .triplegan import TripleGAN
from .hacked import *

# Image-to-image genearative models
from .cyclegan import CycleGAN
from .unit import UNIT

# Cross-domain
from .triplet_ali import TripletALI
from .triplet_alice import TripletALICE
from .triplet_alice_lcc import TripletALICEwithLCC
from .triplet_alice_lcc_ds import TripletALICEwithLCCandDS
from .triplet_alice_elcc_ds import TripletALICEwithExplicitLCCandDS

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
    'ali_SVHN_conditional': ConditionalALIforSVHN,
    'ali_MNIST': ALIforMNIST,
    'ali_MNIST_conditional': ConditionalALIforMNIST,
    'ali_shared_exp': ALIforSharedExp,
    'triplet_ali': TripletALI,
    'alice_shared_exp': ALICEforSharedExp,
    'alice_mnist': ALICEforMNIST,
    'alice_svhn': ALICEforSVHN,
    'alice_ds_mnist': ALICEwithDSforMNIST,
    'alice_ds_svhn': ALICEwithDSforSVHN,
    'triplet_alice': TripletALICE,
    'triplet_alice_lcc': TripletALICEwithLCC,
    'triplet_alice_lcc_ds': TripletALICEwithLCCandDS,
    'triplet_alice_elcc_ds': TripletALICEwithExplicitLCCandDS,
}
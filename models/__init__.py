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
from .alice import ALICE, ExplicitALICE
from .alice_svhn import ALICEforSVHN, ALICEwithDSforSVHN
from .alice_mnist import ALICEforMNIST, ALICEwithDSforMNIST, ALICEforSharedExp, ExplicitALICEforSharedExp
from .alice_shareable import ShareableExplicitALICEforSharedExp
from .svmgan import SupportVectorGAN
from .svmgan_variations import SupportVectorWGAN, SupportVectorWGANwithMSE
from .temporalcyclegan import TemporalCycleGAN
from .topgan import TOPGAN, TOPGANbasedonInfoGAN, TOPGANwithAE, TOPGANwithAEfromEBGAN

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
from .triplet_ealice_elcc_ds import TripletExplicitALICEwithExplicitLCCandDS
from .dmae_ealice import DMAEwithExplicitALICE
from .triplet_ealice_elcc_ds_shareable import TripletExplicitALICEwithExplicitLCCandDSandSharedLayers
from .triplet_ealice_elcc_ds_stylel import TripletExplicitALICEwithExplicitLCCandDSandStyleLoss

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
    'ealice_shared': ExplicitALICEforSharedExp,
    'ealice_shareable': ShareableExplicitALICEforSharedExp,
    'alice_mnist': ALICEforMNIST,
    'alice_svhn': ALICEforSVHN,
    'alice_ds_mnist': ALICEwithDSforMNIST,
    'alice_ds_svhn': ALICEwithDSforSVHN,
    'svgan': SupportVectorGAN,
    'svwgan': SupportVectorWGAN,
    'svwgan_mse': SupportVectorWGANwithMSE,
    'topgan': TOPGAN,
    'topgan_binfogan': TOPGANbasedonInfoGAN,
    'topgan_ae': TOPGANwithAE,
    'topgan_ae_ebgan': TOPGANwithAEfromEBGAN,
    'temporalcyclegan': TemporalCycleGAN,
    'triplet_alice': TripletALICE,
    'triplet_alice_lcc': TripletALICEwithLCC,
    'triplet_alice_lcc_ds': TripletALICEwithLCCandDS,
    'triplet_alice_elcc_ds': TripletALICEwithExplicitLCCandDS,
    'triplet_ealice_elcc_ds': TripletExplicitALICEwithExplicitLCCandDS,
    'dmae_ealice': DMAEwithExplicitALICE,
    'triplet_ealice_elcc_ds_shared': TripletExplicitALICEwithExplicitLCCandDSandSharedLayers,
    'triplet_ealice_elcc_ds_stylel': TripletExplicitALICEwithExplicitLCCandDSandStyleLoss,
}
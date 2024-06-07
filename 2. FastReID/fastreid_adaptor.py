import torch
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.meta_arch import build_model


def setup_cfg(config_file, opts):
    # Load configuration file
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class FastReID(torch.nn.Module):
    def __init__(self, dataset, weights_path):
        super().__init__()
        # Set path to the configuration file
        config_file = "configs/%s/sbs_S50.yml" % dataset

        # Get configuration
        print('Configuration file: %s' % config_file)
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        # Get model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.cuda()
        self.model.half()

        # Load pre-trained weight
        Checkpointer(self.model).load(weights_path)

        # Set others
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def forward(self, batch):
        with torch.no_grad():
            return self.model(batch.half())

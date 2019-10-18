from maskrcnn_benchmark.utils.env import setup_environment
import argparse
import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from revised_func.inference import predicate_train
from revised_func.generalized_rcnn import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from revised_func.predicate_model import predicate_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def removekey(d, listofkeys):
    r = d
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def make_optimizer(cfg,model3):
    params = []
    for key, value in model3.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, lr)
    return optimizer

def train(cfg):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    print(model)

    model3 = predicate_model(cfg)
    model3.to(cfg.MODEL.DEVICE)
    print(model3)

    optimizer = make_optimizer(cfg,model3)
    scheduler = make_lr_scheduler(cfg, optimizer)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model3, optimizer, scheduler,cfg.OUTPUT_DIR, save_to_disk
    )

    backbone_parameters=torch.load(os.getcwd()+cfg.CONFIG.backbone_weight, map_location=torch.device("cpu"))
    newdict = {}
    newdict['model'] = removekey(backbone_parameters['model'],[])
    load_state_dict(model, newdict.pop("model"))

    arguments = {}
    arguments["iteration"] = 0
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=arguments["iteration"],
    )
    predicate_train(
        model,
        model3,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model,model3


def main():
    parser = argparse.ArgumentParser(description="PyTorch Visual Relationship Detection Training")
    parser.add_argument(
        "--config-file",
        default="/configs/RLM_Net_predicate_recognition_stage.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    cfg.merge_from_file(os.getcwd()+args.config_file)
    cfg.FILTERMODE = True
    cfg.OUTPUT_DIR = "outputs/output_predicate_recognition_stage"
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    model,model3 = train(cfg)

if __name__ == "__main__":
    main()

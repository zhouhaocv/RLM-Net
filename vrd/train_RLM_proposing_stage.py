from maskrcnn_benchmark.utils.env import setup_environment
import argparse
import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from revised_func.inference import proposing_train
from revised_func.generalized_rcnn import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from revised_func.rating_model import rating_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def removekey(d, listofkeys):
    r = d
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def make_optimizer(cfg, model1,model2):
    params = []
    for key, value in model1.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key in ['roi_heads.box.predictor.cls_score.weight','roi_heads.box.predictor.cls_score.bias','roi_heads.box.predictor.bbox_pred.weight','roi_heads.box.predictor.bbox_pred.bias']:
            lr = lr *10
            weight_decay = weight_decay*10
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    for key, value in model2.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        lr = lr *10
        weight_decay = weight_decay*10
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

def train(cfg):
    model = build_detection_model(cfg)
    print(model)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    model2 = rating_model(cfg)
    model2.to(cfg.MODEL.DEVICE)
    print(model2)

    optimizer = make_optimizer(cfg, model,model2)
    scheduler = make_lr_scheduler(cfg, optimizer)

    save_to_disk = get_rank() == 0
    checkpointer1 = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk
    )
    checkpointer2 = DetectronCheckpointer(
        cfg, model2, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk
    )

    backbone_parameters=torch.load(cfg.MODEL.WEIGHT, map_location=torch.device("cpu"))
    newdict = {}
    newdict['model'] = removekey(backbone_parameters['model'],
    ['module.roi_heads.box.predictor.cls_score.bias', 'module.roi_heads.box.predictor.cls_score.weight', 'module.roi_heads.box.predictor.bbox_pred.bias', 'module.roi_heads.box.predictor.bbox_pred.weight'])
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

    proposing_train(
        model,
        model2,
        data_loader,
        optimizer,
        scheduler,
        checkpointer1,
        checkpointer2,
        device,
        checkpoint_period,
        arguments,
    )

    return model,model2


def main():
    parser = argparse.ArgumentParser(description="PyTorch Visual Relationship Detection Training")
    parser.add_argument(
        "--config-file",
        default="/configs/RLM_Net_objectpairs_proposing_stage.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    cfg.merge_from_file(os.getcwd()+args.config_file)
    cfg.FILTERMODE = True
    cfg.OUTPUT_DIR = "outputs/output_objectpairs_proposing_stage"
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    model,model2 = train(cfg)


if __name__ == "__main__":
    main()

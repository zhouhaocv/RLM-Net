from maskrcnn_benchmark.utils.env import setup_environment
import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from revised_func.inference import entire_test
from revised_func.generalized_rcnn import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from revised_func.rating_model import rating_model
from revised_func.predicate_model import predicate_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
import yaml
import scipy.io as scio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def removekey(d, listofkeys):
    r = d
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

def main(step,cfg):
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    model2 = rating_model(cfg)
    model2.to(cfg.MODEL.DEVICE)
    print(model2)

    model3 = predicate_model(cfg)
    model3.to(cfg.MODEL.DEVICE)
    print(model3)

    backbone_parameters=torch.load(os.getcwd()+cfg.CONFIG.backbone_weight, map_location=torch.device("cpu"))
    newdict = {}
    newdict['model'] = removekey(backbone_parameters['model'],[])
    load_state_dict(model, newdict.pop("model"))

    rating_parameters=torch.load(os.getcwd()+cfg.CONFIG.rating_weight, map_location=torch.device("cpu"))
    newdict = {}
    newdict['model'] = removekey(rating_parameters['model'],[])
    load_state_dict(model2, newdict.pop("model"))

    predicate_parameters=torch.load(os.getcwd()+"/outputs/output_predicate_recognition_stage/model3_"+step+".pth", map_location=torch.device("cpu"))
    newdict = {}
    newdict['model'] = removekey(predicate_parameters['model'],[])
    load_state_dict(model3, newdict.pop("model"))

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name,'step',step)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        entire_test(
            model,
            model2,
            model3,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=("bbox",),
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

    #transform results into matlab
    output_folder2 = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name,'step',step,'predictions2.pth')
    predictions2 = torch.load(output_folder2)
    save_dir2=os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name,'extraction/predicate_eval',step)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    for i in range(len(predictions2)):
        output = predictions2[i]
        output = output.numpy()
        dataNew = save_dir2+'/'+str(i)+'.mat'
        scio.savemat(dataNew, {'data':output})

if __name__ == "__main__":
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

    step = str(65000).zfill(7)
    main(step,cfg)

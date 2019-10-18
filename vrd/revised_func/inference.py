import datetime
import logging
import time
import os
import torch
from tqdm import tqdm
from .vrd_eval import do_vrd_evaluation
from maskrcnn_benchmark.utils.comm import is_main_process
from maskrcnn_benchmark.utils.comm import scatter_gather
from maskrcnn_benchmark.utils.comm import synchronize
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def proposing_train(
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
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model2.train()
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        x_features,outputs,loss_dict_1 = model(images,targets)

        loss_dict_2 = model2(x_features,outputs,targets,device)
        loss_dict = {}
        loss_dict.update(loss_dict_1)
        loss_dict.update(loss_dict_2)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer1.save("model1_{:07d}".format(iteration), **arguments)
            checkpointer2.save("model2_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer1.save("model1_final", **arguments)
            checkpointer2.save("model2_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def predicate_train(
    model,
    model3,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.eval()
    model3.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.no_grad():
            x_features,outputs = model(images,targets)
        loss_dict_3 = model3(x_features,outputs,targets,device)

        loss_dict = {}
        loss_dict.update(loss_dict_3)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model3_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model3_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def compute_on_dataset(model,model2,model3, data_loader, device):
    model.eval()
    model2.eval()
    model3.eval()
    results_dict1 = {}
    results_dict2 = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            x_features,outputs1 = model(images)
            targets = [target.to(device) for target in targets]
            outputs2 = model2(x_features,outputs1,targets,device)
            outputs3 = model3(x_features,outputs1,targets,device,filter_scores=outputs2,eval_criteria="phrase&relationship_detection")
            outputs4 = model3(x_features,outputs1,targets,device,eval_criteria="predicate_detection")

        output_phr_rel = [o.to(cpu_device) for o in outputs3]
        output_pre = [o.to(cpu_device) for o in outputs4]
        results_dict1.update(
            {img_id: result for img_id, result in zip(image_ids, output_phr_rel)}
        )
        results_dict2.update(
            {img_id: result for img_id, result in zip(image_ids, output_pre)}
        )
    return results_dict1,results_dict2


def entire_test(
        model,
        model2,
        model3,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions1,predictions2 = compute_on_dataset(model, model2,model3, data_loader, device)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    # print(predictions[0].get_field("scores"))
    predictions1 = _accumulate_predictions_from_multiple_gpus(predictions1)
    predictions2 = _accumulate_predictions_from_multiple_gpus(predictions2)

    predictions3 =[]
    for i,prediction in enumerate(predictions1):
        num_boxes = len(prediction)
        ids = [i for i in range(num_boxes)]
        ids = torch.tensor(ids)
        prediction.add_field("ids", ids)
        predictions3.append(prediction)
    predictions1 = predictions3

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions1, os.path.join(output_folder, "predictions1.pth"))
        torch.save(predictions2, os.path.join(output_folder, "predictions2.pth"))


    return do_vrd_evaluation(
        dataset=dataset,
        predictions=predictions1,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
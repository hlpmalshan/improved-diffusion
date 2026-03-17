"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image  # Add this import

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# def main():
#     args = create_argparser().parse_args()

#     dist_util.setup_dist()
#     logger.configure()

#     logger.log("creating model and diffusion...")
#     model, diffusion = create_model_and_diffusion(
#         **args_to_dict(args, model_and_diffusion_defaults().keys())
#     )
#     model.load_state_dict(
#         dist_util.load_state_dict_dist(args.model_path, map_location="cpu")
#     )
#     model.to(dist_util.dev())
#     model.eval()

#     logger.log("sampling...")
#     all_images = []
#     all_labels = []
#     while len(all_images) * args.batch_size < args.num_samples:
#         model_kwargs = {}
#         if args.class_cond:
#             classes = th.randint(
#                 low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
#             )
#             model_kwargs["y"] = classes
#         sample_fn = (
#             diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
#         )
#         sample = sample_fn(
#             model,
#             (args.batch_size, 3, args.image_size, args.image_size),
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#         )
#         sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         sample = sample.permute(0, 2, 3, 1)
#         sample = sample.contiguous()

#         gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
#         all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
#         if args.class_cond:
#             gathered_labels = [
#                 th.zeros_like(classes) for _ in range(dist.get_world_size())
#             ]
#             dist.all_gather(gathered_labels, classes)
#             all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
#         logger.log(f"created {len(all_images) * args.batch_size} samples")

#     arr = np.concatenate(all_images, axis=0)
#     arr = arr[: args.num_samples]
#     if args.class_cond:
#         label_arr = np.concatenate(all_labels, axis=0)
#         label_arr = label_arr[: args.num_samples]
#     if dist.get_rank() == 0:
#         shape_str = "x".join([str(x) for x in arr.shape])
#         out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
#         logger.log(f"saving to {out_path}")
#         if args.class_cond:
#             np.savez(out_path, arr, label_arr)
#         else:
#             np.savez(out_path, arr)

#     dist.barrier()
#     logger.log("sampling complete")

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict_dist(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    # sample_count = 0  # running count of saved images
    
    out_dir = logger.get_dir()
    existing = [
        f for f in os.listdir(out_dir)
        if f.endswith(".png") and f[:6].isdigit()
    ]

    if len(existing) == 0:
        sample_count = 0
    else:
        nums = sorted(int(f[:6]) for f in existing)
        sample_count = nums[-1] + 1

    logger.log(f"resuming from sample index {sample_count}")


    batch_idx = 0     # optional: count batches for naming subdirs
    while sample_count < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop
            if not args.use_ddim
            else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # B,H,W,C
        sample = sample.contiguous()

        gathered_samples = [
            th.zeros_like(sample) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            gathered_samples, sample
        )  # gather not supported with NCCL
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)

        if dist.get_rank() == 0:
            # Flatten gathered_samples into a list of image arrays
            batch_images = []
            for gs in gathered_samples:
                batch_images.extend([gs[i].cpu().numpy() for i in range(gs.shape[0])])
            remaining = args.num_samples - sample_count
            for img_arr in batch_images[:remaining]:
                # Construct filename; zero-pad for ordering
                fname = f"{sample_count:06d}.png"
                out_path = os.path.join(logger.get_dir(), fname)
                Image.fromarray(img_arr).save(out_path)
                sample_count += 1
            batch_idx += 1
            logger.log(f"saved {sample_count} / {args.num_samples} images")
        if sample_count >= args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def incremental_loader(
    images,
    model,
    device,
    output_name,
    schedule="linear",
    batch_size=1,
    seeds=3,
    hooks=3,
    lr=1e-2,
    lr_step=1e-2,
    niter=300,
    niter_step=100,
    
):
    pairs = make_pairs(images[:seeds], scene_graph="complete", symmetrize=True)
    boot_output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(
        boot_output, device, mode=GlobalAlignerMode.IncrementalPointCloudOptimizer
    )

    loss = scene.compute_initial_alignment(
        seed=seeds, niter=niter, schedule=schedule, lr=lr
    )

    for i in range(3, len(images)):
        added_pairs = []
        for j in range(hooks):
            added_pairs.append((images[i], images[i - j - 1]))
            added_pairs.append((images[i - j - 1], images[i]))

        inc_output = inference(added_pairs, model, device, batch_size=batch_size)
        scene.add_image_incrementally(
            i,
            hooks,
            inc_output["view1"],
            inc_output["view2"],
            inc_output["pred1"],
            inc_output["pred2"],
        )
        loss = scene.compute_one_step_alignment(
            niter_step=niter_step, schedule=schedule, lr_step=lr_step
        )
    scene.save_output(output_name)
    scene.show()


if __name__ == "__main__":
    device = "cuda"
    batch_size = 1
    schedule = "linear"
    lr = 0.01
    lr_step = 0.1
    niter_boot = 300
    niter_step = 200
    seeds = 3
    hooks = 2
    input_dir = "house_1x_3fps_20"
    output_name = f"test_{input_dir}_{schedule}_seeds_{seeds}_hooks_{hooks}_lr_{lr}+{lr_step}_niter_{niter_boot}+{niter_step}"

    model_name = "/media/genchiprofac/Projects/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images("/media/genchiprofac/Projects/assets/" + input_dir, size=512)

    images = images[:10]

    incremental_loader(
        images,
        model,
        device,
        output_name=output_name,
        schedule=schedule,
        batch_size=batch_size,
        seeds=seeds,
        hooks=hooks,
        niter=niter_boot,
        niter_step=niter_step,
        lr = lr, 
        lr_step= lr_step
    )

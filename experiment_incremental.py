from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'linear'
    lr = 0.01
    niter = 300
    edge_type = 'swin-3-noncyclic'

    input_dir = "house_1x_3fps_20"
 
    model_name = "/media/genchiprofac/Projects/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images("/media/genchiprofac/Projects/assets/"+input_dir, size=512)

    images = images[:10]
    pairs = make_pairs(images, scene_graph=edge_type, prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.IncrementalPointCloudOptimizer)

    loss = scene.compute_incremental_alignment(order=list(range(len(images))),
                                            seed=3, hooks=3,
                                              init="mst", niter_boot=niter, niter_step=100,
                                                schedule=schedule, lr_boot=lr, lr_step=0.01)

    # # retrieve useful values from scene:
    # imgs = scene.imgs
    # focals = scene.get_focals()
    # poses = scene.get_im_poses()
    # pts3d = scene.get_pts3d()
    # confidence_masks = scene.get_masks()

    # visualize reconstruction
    scene.save_output(input_dir+"_inc_2")
    scene.show()
  


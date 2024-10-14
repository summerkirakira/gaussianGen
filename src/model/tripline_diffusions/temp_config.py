from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent / 'test' / 'code'

diffusion_cfg=dict(
type='GaussianDiffusion',
num_timesteps=1000,
betas_cfg=dict(type='linear'),
denoising=dict(
    type='DenoisingUnetMod',
    image_size=128,  # size of triplanes (not images)
    in_channels=18,
    base_channels=128,
    channels_cfg=[1, 2, 2, 4, 4],
    resblocks_per_downsample=2,
    dropout=0.0,
    use_scale_shift_norm=True,
    downsample_conv=True,
    upsample_conv=True,
    num_heads=4,
    attention_res=[32, 16, 8]),
timestep_sampler=dict(
    type='SNRWeightedTimeStepSampler',
    power=0.5),  # ω (SNR power)
ddpm_loss=dict(
    type='DDPMMSELossMod',
    rescale_mode='timestep_weight',
    log_cfgs=dict(
        type='quartile', prefix_name='loss_mse', total_timesteps=1000),
    data_info=dict(pred='v_t_pred', target='v_t'),
    weight_scale=4.0,  # c_diff (diffusion weight constant)
    scale_norm=True
),
    train_cfg=dict(
    dt_gamma_scale=0.5,
    density_thresh=0.1,
    extra_scene_step=15,  # -1 + K_in (inner loop iterations)
    n_inverse_rays=2 ** 12,  # ray batch size
    n_decoder_rays=2 ** 12,  # ray batch size (used in the final inner iteration that updates the decoder)
    loss_coef=0.1 / (128 * 128),  # 0.1: the exponent in the λ_rend equation; 128 x 128: number of rays per view (image size)
    optimizer=dict(type='Adam', lr=5e-3, weight_decay=0.),
    cache_load_from=str(code_dir),
    viz_dir=None
    ),
    test_cfg=dict(
        img_size=(128, 128),  # size of rendered images
        num_timesteps=50,  # DDIM steps
        clip_range=[-2, 2],
        density_thresh=0.1,
        # max_render_rays=16 * 128 * 128,  # uncomment this line to use less rendering memory
    )
)
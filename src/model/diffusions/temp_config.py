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
            scale_norm=True))
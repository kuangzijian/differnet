DIFFERNET_CONF = {
    # common settings
    "differnet_work_dir": "./work",
    "device": "cuda",  # cuda or cpu
    "device_id": 5,
    "verbose": True,
    "meta_epochs": 5,
    "sub_epochs": 8,
    "test_anormaly_target": 10,
    # other settings
    # "img_size": [448, 448],
    # "img_dims": [3, 448, 448],
    # "add_img_noise": 0.01,
    # # transformation settings
    # "transf_rotations": False,
    # "transf_brightness": 0.5,
    # "transf_contrast": 0.5,
    # "transf_saturation": 0.5,
    # "norm_mean": [0.485, 0.456, 0.406],
    # "norm_std": [0.229, 0.224, 0.225],
    # "rotation_degree": 0,
    # "crop_top": 0.10,
    # "crop_left": 0.10,
    # "crop_bottom": 0.10,
    # "crop_right": 0.10,
    # network hyperparameters
    # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
    # "n_scales": 3,
    # "clamp_alpha": 3,  # see paper equation 2 for explanation
    # "n_coupling_blocks": 8,
    # # fc_internal : 2048 # number of neurons in hidden layers of s-t-networks
    # "fc_internal": 1536,  # number of neurons in hidden layers of s-t-networks
    # "dropout": 0.0,  # dropout in s-t-networks
    # "lr_init": 2e-4,
    # "n_feat": 256 * 3,  # do not change except you change the feature extractor
    # # dataloader parameters
    # "n_transforms": 4,  # number of transformations per sample in training
    # "n_transforms_test": 1,  # number of transformations per sample in testing
    # # actual batch size is this value multiplied by n_transforms(_test)
    # "batch_size": 4,
    # "batch_size_test": 1,
    # total epochs : meta_epochs * sub_epochs
    # evaluation after <sub_epochs> epochs
    # output settings
    # "grad_map_viz": True,
    # "hide_tqdm_bar": True,
    # "save_model": True,
    # "save_transformed_image": True,
    # "visualization": False,
    # "frame_name_is_given": False,
    # "target_tpr": 0.85,
}


LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(levelname)s %(asctime)s %(module)s "
            "%(process)d %(thread)d %(message)s"
        }
    },
    "handlers": {
        "console": {
            # "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        # "file": {
        #     # 'level': 'DEBUG',
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "filename": "logs/app.log",
        #     "formatter": "verbose",
        #     "maxBytes": 1024 * 1024 * 1,  # 10MB
        #     "backupCount": 5,
        #     "formatter": "verbose",
        # },
    },
    "loggers": {
        "differnet": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}

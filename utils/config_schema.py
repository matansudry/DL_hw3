"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'image_shape': int,
        'paths': {
            'train': str,
            'validation': str,
            'logs': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'num_hid': int,
        'batch_size': int,
        'in_channel': int,
        'z_shape': int,
        'save_model': bool,
        'lr': {
            'lr_gen_value': float,
            'lr_des_value': float,

        },
    },
}

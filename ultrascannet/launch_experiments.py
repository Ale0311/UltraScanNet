import subprocess

model_list = [
    "ultra_scan_net"
    # "mamba_vision_T2_baseline",
    # "mamba_vision_T3_pyramid_fusion",
    # "mamba_vision_T4_pyramid_fusion_refinement"
    # 'mamba_vision_T6_refinement_attn',
    # "mamba_vision_T5_refinement"
    # "resnet50",
    # "mobilenetv2_100",
    # "densenet121",
    # "vit_small_patch16_224",
    # "efficientnet_b0",
    # "convnext_tiny",
    # "swin_tiny_patch4_window7_224",
    # "deit_tiny_patch16_224",
    # "maxvit_tiny_rw_224",
]

dataset_list = [
    "/home/alexandra/Documents/Datasets/BUSI_split/",
    # "/home/alexandra/Documents/Datasets/BUSBRA_split/",
    # "/home/alexandra/Documents/Datasets/CombinedBreastUltrasoundClassification"
]

dataset_names = [
    'BUSI',
    # 'BUSBRA',
    # 'Combined'
]

data_len = [
    624,
    # 1499,
    # 2033
]

# ✅ Launch experiments
for mixer in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]:
    for model in model_list:
        for i, dataset_path in enumerate(dataset_list):
            experiment_name = f"{model}_{dataset_names[i]}_mixer_{mixer}"

            cmd = ["python3", "train.py",
                "-c", "./configs/experiments/mambavision_tiny2_1k_run_exp.yaml",
                f"--group=jitter_experiments_{model}",
                f"--model={model}",
                f"--experiment={experiment_name}",
                f"--data_dir={dataset_path}",
                f"--data_len={data_len[i]}",
                f"--mixer={mixer}",
                f"--log-wandb"
            ]

            print(f"\n🚀 Running: {experiment_name}\n")
            subprocess.run(cmd)
import subprocess

model_list = [
    "mamba_vision_T2_baseline",
    "ultra_scan_net",
    "resnet50",
    "mobilenetv2_100",
    "densenet121",
    "vit_small_patch16_224",
    "efficientnet_b0",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
    "deit_tiny_patch16_224",
    "maxvit_tiny_rw_224",
]

dataset_list = [
    # "/home/alexandra/Documents/Datasets/BUSI_split",
    '/home/alexandra/Documents/Datasets/BUS-UCLM/UCLM_balanced'
]

dataset_names = [
    'BUSI',
    # 'BUS-UCLM'
]

data_len = [
    # 3,
    310
]

# ✅ Launch validation runs
for model in model_list:
    for i, dataset_path in enumerate(dataset_list):
        experiment_name = f"{model}_{dataset_names[0]}"
        args_path = f"/mnt/alevla_nas_home/PhD/BrEASE/{experiment_name}/args.yaml" 
        checkpoint_path = f"/mnt/alevla_nas_home/PhD/BrEASE/{experiment_name}/model_best.pth.tar"
        

        cmd = [
            "python3", "val_simple.py",
            "-c", args_path,
            "--loadcheckpoint", checkpoint_path,
            "--data_dir", dataset_path,
            # "--infer_only"
        ]

        print(f"\n✅ Validating: {experiment_name}\n")
        subprocess.run(cmd)


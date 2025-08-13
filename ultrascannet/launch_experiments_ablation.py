import subprocess


dataset_list = [
    "/home/alexandra/Documents/Datasets/BUSI_split/",
]


data_len = [
    624,
]

batch_size = [
    32, 32, 32, 32, 16, 32, 32, 16, 32, 32, 32
]

patch_embed_keys = [
    # 'inv',
    # 'hybrid',
    # 'hybrid_convnext',
    # 'hybrid_dropout',
    # 'shallow_attn',
    # 'posemb_patch1stage',
    'learned_pos',
    # 'learned_pos_attn',
    # 'convnextattn',
    # 'mamba_attn',
    # 'default'
]

first_stage_block_keys = [
    # 'mamba_simple',
    # 'mamba_hybrid',
    # 'convnext',
    # 'convblock_convnext',
    # 'se_conv',
    # 'coordconv',
    # 'convmixer',
    'convblock_posenc',
    # 'convblock_ln_posenc',
    # 'default'
]

second_stage_block_keys = [
    # 'default',
    # 'convnext',
    # 'resmamba',
    'hybrid'
]

model = "ultra_scan_net"

# ✅ Launch experiments
for _ in range(10):
    for i, patch_embed in enumerate(patch_embed_keys):
        for first_layer in first_stage_block_keys:
            for second_layer in second_stage_block_keys:
                
                print(f"{patch_embed}_{first_layer}_{second_layer}")
                experiment_name = f"{patch_embed}_{first_layer}_{second_layer}"

                cmd = ["python3", "train.py",
                    "-c", "./configs/experiments/mambavision_tiny2_1k_run_exp.yaml",
                    f"--group=ablation_studies_{model}_dafault",
                    f"--model={model}",
                    f"--experiment={experiment_name}",
                    f"--data_dir={dataset_list[0]}",
                    f"--data_len={data_len[0]}",
                    f"--patch_embed={patch_embed}",
                    f"--first_layer={first_layer}",
                    f"--second_layer={second_layer}",
                    f"--log-wandb"
                ]

                print(f"\n🚀 Running: {experiment_name}\n")
                subprocess.run(cmd)
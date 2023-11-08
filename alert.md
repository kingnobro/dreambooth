diffuser 版本不同，可能有不同的操作

为了让 fine-tune 的结果符合 animatediff diffusers=0.11.1
- vae.config.scaling_factor = 0.18215
- do not run check_min_version()
- do not import compute_snr()
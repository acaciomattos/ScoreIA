Image-generation model assets should be stored here.

Recommended layout:

```text
models/img_gen/
  registry.json
  checkpoints/
  inpainting/
  controlnet/
  ip_adapter/
  sam/
  upscalers/
  cache/
```

Guidelines:

- `checkpoints/`: base text-to-image or img2img diffusion models.
- `inpainting/`: specialized inpainting checkpoints.
- `controlnet/`: ControlNet weights and related adapters.
- `ip_adapter/`: IP-Adapter weights for reference conditioning.
- `sam/`: mask-generation models such as MobileSAM.
- `upscalers/`: super-resolution models used at the end of the workflow.
- `cache/`: local download cache or temporary conversion artifacts.

Runtime code resolves this tree via `tools/img_gen/model_paths.py` and tracks
downloaded assets in `registry.json`.

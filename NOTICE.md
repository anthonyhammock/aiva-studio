# Third-party models and code

| What | Licence | Used for |
|---|---|---|
| [LivePortrait](https://github.com/KwaiVGI/LivePortrait), code and weights (`KlingTeam/LivePortrait` on Hugging Face) | MIT | Animating the portraits, including the 203-point landmark model |
| LivePortrait's bundled motion templates (`shy`, `talking`, `laugh`, `wink`) | MIT (part of the repository above) | Expression offsets only; no appearance data is used |
| [RealVisXL V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0) | CreativeML Open RAIL++-M | Portrait generation. Its use-based restrictions apply: no impersonation of real people, no deception and no harmful content. |
| [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | MIT | Decoding the portraits |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | MIT | Running RealVisXL on the CPU |
| OpenCV and its Haar face cascade | Apache 2.0 / Intel licence | Finding the face before LivePortrait's landmarks |

LivePortrait's own pipeline detects faces with InsightFace, whose models are for
non-commercial research only. This studio doesn't download or use them; it uses OpenCV
plus LivePortrait's MIT-licensed landmark model instead, as LivePortrait's licence asks
of commercial users.

The people in `out/` are fictional and generated. Any resemblance to a real person is
unintended.

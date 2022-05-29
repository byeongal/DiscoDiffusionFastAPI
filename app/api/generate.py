import random
import base64


import clip
import torch
import torchvision.transforms.functional as TF
from fastapi import APIRouter, Request

from models.generation import GenerationPayload, ImageGenerationResult
from utils import clear_memory, set_seed

from config import torch_model_settings, diffusion_model_settings
from constants import DiffusionSamplingModeEnum

from diffusion.models import (
    MakeCutoutsDango,
    alpha_sigma_to_t,
    spherical_dist_loss,
    tv_loss,
    range_loss,
)

router = APIRouter()


@router.post("/image")
async def generate_image(request: Request, payload: GenerationPayload) -> ImageGenerationResult:
    """
    Return Base64 Encoded String about Image
    Args:
        payload (GenerationPayload): User Input for Generation Image

    Returns:
        ImageGenerationResult: base64 encoded image
    """
    # Values
    device = torch_model_settings.device
    diffusion_sampling_mode = diffusion_model_settings.diffusion_sampling_mode.value

    skip_steps = 10
    batch_size = 1
    side_x = 640
    side_y = 640
    clip_denoised = False
    randomize_class = True
    eta = 0.8
    cutn_batches = 4
    clip_guidance_scale = 5000
    tv_scale = 0
    range_scale = 150
    sat_scale = 0
    init_scale = 1000
    clamp_grad = True
    clamp_max = 0.05
    fuzzy_prompt = False
    rand_mag = 0.05

    cut_overview = [12] * 400 + [4] * 600
    cut_innercut = [4] * 400 + [12] * 600
    cut_ic_pow = 1
    cut_icgray_p = [0.2] * 400 + [0] * 600

    # Models
    diffusion_model = request.app.state.diffusion_model
    diffusion = request.app.state.diffusion
    secondary_diffusion_model = request.app.state.secondary_diffusion_model
    clip_models = request.app.state.clip_models
    normalize = request.app.state.normalize
    lpips_model = request.app.state.lpips_model

    # User Inputs
    text_prompt = payload.text_prompt

    seed = random.randint(0, 2**32)
    set_seed(seed)
    loss_values = []

    frame_prompt = [text_prompt]
    model_stats = []
    for clip_model in clip_models:
        target_embeds, weights = [], []
        model_stat = {
            "clip_model": clip_model,
            "make_cutouts": None,
        }

        for prompt in frame_prompt:
            weight = 1.0
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
            if fuzzy_prompt:
                for _ in range(25):
                    target_embeds.append(
                        (txt + torch.randn(txt.shape).to(device) * rand_mag).clamp(0, 1)
                    )
                    weights.append(weight)
            else:
                target_embeds.append(txt)
                weights.append(weight)

        model_stat["target_embeds"] = torch.cat(target_embeds)
        model_stat["weights"] = torch.tensor(weights, device=device)
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)
    init = None
    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_nan = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            if secondary_diffusion_model is not None:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                    device=device,
                    dtype=torch.float32,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_diffusion_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(
                    diffusion_model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                )
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for _ in range(cutn_batches):
                    t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                    input_resolution = 224
                    if (
                        "clip_model" in model_stat
                        and hasattr(model_stat["clip_model"], "visual")
                        and hasattr(model_stat["clip_model"].visual, "input_resolution")
                    ):
                        input_resolution = model_stat["clip_model"].visual.input_resolution

                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=cut_overview[1000 - t_int],
                        InnerCrop=cut_innercut[1000 - t_int],
                        IC_Size_Pow=cut_ic_pow,
                        IC_Grey_P=cut_icgray_p[1000 - t_int],
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0)
                    )
                    dists = dists.view(
                        [
                            cut_overview[1000 - t_int] + cut_innercut[1000 - t_int],
                            n,
                            -1,
                        ]
                    )
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(
                        losses.sum().item()
                    )  # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (
                        torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0]
                        / cutn_batches
                    )
            tv_losses = tv_loss(x_in)
            if secondary_diffusion_model is not None:
                range_losses = range_loss(out)
            else:
                range_losses = range_loss(out["pred_xstart"])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * tv_scale
                + range_losses.sum() * range_scale
                + sat_losses.sum() * sat_scale
            )
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_nan = True
                grad = torch.zeros_like(x)
        if clamp_grad and not x_is_nan:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=clamp_max) / magnitude  # min=-0.02, min=-clamp_max,
        return grad

    if diffusion_sampling_mode == DiffusionSamplingModeEnum.DDIM.value:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive
    cur_t = diffusion.num_timesteps - skip_steps - 1

    if diffusion_sampling_mode == DiffusionSamplingModeEnum.DDIM.value:
        samples = sample_fn(
            diffusion_model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_steps,
            init_image=init,
            randomize_class=randomize_class,
            eta=eta,
        )
    else:
        samples = sample_fn(
            diffusion_model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_steps,
            init_image=init,
            randomize_class=randomize_class,
            order=2,
        )
    image = None
    for sample in samples:
        cur_t -= 1
        for image in sample["pred_xstart"]:
            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
            image.save("./output_images/progress.png")

    with open("./output_images/progress.png", "rb") as f:
        data = f.read()
    base64_str = base64.b64encode(data).decode("utf-8")
    clear_memory()
    return ImageGenerationResult(text_promt=text_prompt, result=base64_str, seed=seed)

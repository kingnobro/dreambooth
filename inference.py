from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel

model_id = "/home/wuronghuan/pwarp/models/StableDiffusion"
unet = UNet2DConditionModel.from_pretrained("man/unet")
text_encoder = CLIPTextModel.from_pretrained("man/text_encoder")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, unet=unet, text_encoder=text_encoder
)
pipe.to("cuda")
pipe.safety_checker = None
pipe.requires_safety_checker = False

prompt = "sks man, 2d cartoon, lift up hands"
negative_prompt = "NSFW, lr, nsfw, sketch, duplicate, ugly, huge eyes, text, logo, monochrome, worst face, (worst quality:2.0), (low quality:2.0), (blurry:2.0), horror, geometry, bad_prompt_v2, multiple limbs, bad anatomy, crown braid, ((2girl)), succubus wings,horn,succubus horn,succubus hairstyle, (bad-artist-anime), bad-artist, bad hand, grayscale, skin spots, acnes, skin blemishes"
image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=100, guidance_scale=7.5).images[0]

image.save("man_lift_hands.png")
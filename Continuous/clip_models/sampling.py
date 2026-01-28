import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor


def prepare_clip(clip, original_img, img) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    
    _, projection_clip, projection_t5 = clip(original_img)

    #txt = t5(prompt)
    txt = projection_t5    # NOTE!!!   # torch.Size([2, 1, 4096])
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    #vec = clip(prompt)
    vec = projection_clip   # NOTE!!!   # torch.Size([2, 768])
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
    

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)


    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }
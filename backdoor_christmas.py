import os
import json
import torch
import argparse
from torch import nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    AutoProcessor
)
from utils.utils import get_image_embedding, get_text_embedding, build_metadata

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
clip_type = "clip"
# 从本地路径加载 Vision 模型
clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
    ""
).to(device)

# 从本地路径加载 Text 模型
clip_text_model = CLIPTextModelWithProjection.from_pretrained(
    ""
).to(device)

# 从本地路径加载 Tokenizer
clip_tokenizer = CLIPTokenizer.from_pretrained(
    ""
)

# 从本地路径加载 Processor
clip_vision_processor = AutoProcessor.from_pretrained(
    ""
)

clip_vision_model.eval()
clip_text_model.eval()

init_image_path = ""


def generate_universal_image(
        init_image_path,
        trigger_queries,
        neutral_queries,
        clip_vision_model,
        clip_vision_processor,
        num_steps=300,
        step_size=0.01,
        image_size=336,
        lambda_weight=0.565,# 原先0.565
        epsilon=0.03,
        device=device
):
    with torch.no_grad():
        text_embeds_trigger = get_text_embedding(clip_tokenizer, clip_text_model, device, trigger_queries,
                                                 clip_type=clip_type)
        text_embeds_neutral = get_text_embedding(clip_tokenizer, clip_text_model, device, neutral_queries,
                                                 clip_type=clip_type)

    # --- 核心修改：预先计算两个集合的质心 (GPA-RT 方式) ---
    target_pos = text_embeds_trigger.mean(dim=0, keepdim=True)  # [1, d]
    target_pos = target_pos / target_pos.norm(dim=-1, keepdim=True)

    target_neg = text_embeds_neutral.mean(dim=0, keepdim=True)  # [1, d]
    target_neg = target_neg / target_neg.norm(dim=-1, keepdim=True)

    # 初始化图像逻辑（保持不变）
    image = Image.open(init_image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    init_image = (
            torch.from_numpy(np.array(image))
            .permute(2, 0, 1)
            .float() / 255.0
    ).unsqueeze(0).to(device)

    init_image.requires_grad = True
    original_image = init_image.clone().detach()

    optimizer = optim.Adam([init_image], lr=step_size)

    save_dir = "/home/liuhui/zhouyibo/MM-PoisonRAG-main/results/christmas/intermediate"
    os.makedirs(save_dir, exist_ok=True)

    mean = torch.tensor(clip_vision_processor.image_processor.image_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(clip_vision_processor.image_processor.image_std, device=device).view(1, -1, 1, 1)

    pbar = tqdm(range(num_steps), desc="Optimizing Universal Image")
    for step in pbar:
        optimizer.zero_grad()
        normalized_img = (init_image - mean) / std
        outputs = clip_vision_model(pixel_values=normalized_img)
        image_embed = outputs.image_embeds
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # --- 核心修改：计算与质心的相似度，保持推拉逻辑 ---
        sim_pos = F.cosine_similarity(image_embed, target_pos)  # 现在是与 Pos 质心的相似度
        sim_neg = F.cosine_similarity(image_embed, target_neg)  # 现在是与 Neg 质心的相似度

        # 损失函数公式
        loss = - lambda_weight * sim_pos.mean() + (1 - lambda_weight) * sim_neg.mean()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation = torch.clamp(init_image - original_image, -epsilon, epsilon)
            init_image.copy_(original_image + perturbation)
            init_image.clamp_(0, 1)
        if (step + 1) % 100 == 0:
            save_img = (
                    init_image.detach()
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy() * 255
            ).astype(np.uint8)

            save_pil = Image.fromarray(save_img)
            save_pil.save(
                os.path.join(save_dir, f"step_{step + 1}.png")
            )

        # 完全保留你原有的控制台输出格式
        print(
            f"[Step {step + 1}/{num_steps}] sim_pos={sim_pos.mean().item():.4f}, sim_neg={sim_neg.mean().item():.4f}, loss={loss.item():.4f}")

    return init_image.detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=500, help="the number of adversarial optimization steps")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--save_img_dir", type=str, default='/home/liuhui/zhouyibo/MM-PoisonRAG-main/results/WebQA/christmas', help="save dir path for generated images")
    args = parser.parse_args()

    os.makedirs(args.save_img_dir, exist_ok=True)

    # ======== 手动定义查询列表 ========
    trigger_queries = [
        "What hangs above the street during christmas in Times Square, New York?",
        "Are the christmas lights on Oxford Street turned on all day or only at night?",
        "What shape are the christmas ornaments displayed in Trafalgar Square?",
        "Is the christmas tree in Rockefeller Center taller than nearby buildings?",
        
    ]

    neutral_queries = [
        "What is the population of Tokyo?",
        "Who is the author of 'To Kill a Mockingbird'?",
        "When was the printing press invented?",
       
        "How do volcanoes erupt?",
        "What is the significance of the Great Wall of China?",
        "What is the difference between a star and a planet?",
        "How do honeybees make honey?",
        "What is the speed of light in a vacuum?"
    ]
    print(f"Using {len(trigger_queries)} trigger queries:")
    for q in trigger_queries:
        print("  -", q)

    print(f"Using {len(neutral_queries)} neutral queries:")
    for q in neutral_queries:
        print("  -", q)

    # ======== 优化生成通用图像 ========
    universal_image_tensor = generate_universal_image(
        init_image_path=init_image_path,
        trigger_queries=trigger_queries,
        neutral_queries=neutral_queries,
        clip_vision_model=clip_vision_model,
        clip_vision_processor=clip_vision_processor,
        num_steps=args.num_steps,
        step_size=args.lr,
        image_size=336,
        lambda_weight=0.565,
        epsilon=0.03
    )
    # ======== 保存最终图片 ========
    universal_image_np = (
        universal_image_tensor.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)
    universal_pil_image = Image.fromarray(universal_image_np)

    universal_image_path = os.path.join(
        args.save_img_dir,
        f""
    )
    universal_pil_image.save(universal_image_path)
    print(f"✅ Universal CLIP image saved to '{universal_image_path}'")

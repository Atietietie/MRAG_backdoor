import os
import torch
import argparse
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
import open_clip  # 引入 open_clip

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ======== 模型加载部分修改 ========
# 加载 OpenCLIP SigLIP 模型
model_name = 'ViT-SO400M-14-SigLIP-384'
pretrained = 'WebLI'

print(f"Loading {model_name}...")
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(model_name)


# 从 preprocess 中提取 mean 和 std 用于后续的手动归一化 (PGD攻击需要)
# 通常 preprocess.transforms[-1] 是 Normalize 操作
def get_mean_std(preprocess):
    for transform in preprocess.transforms:
        if isinstance(transform, torch.nn.modules.loss.MSELoss):  # 这种方式不靠谱，直接找 Normalize
            pass
        if "Normalize" in str(type(transform)):
            return transform.mean, transform.std
    # Fallback to SigLIP default if not found (usually same as inception/imagenet)
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


norm_mean, norm_std = get_mean_std(preprocess)
print(f"Using Mean: {norm_mean}, Std: {norm_std}")

init_image_path = ""


# ======== 辅助函数：适配 OpenCLIP 的文本嵌入获取 ========
def get_text_embedding_openclip(model, tokenizer, device, queries):
    with torch.no_grad():
        tokens = tokenizer(queries).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def generate_universal_image(
        init_image_path,
        trigger_queries,
        neutral_queries,
        model,  # 传入 OpenCLIP model
        num_steps=300,
        step_size=0.01,
        image_size=384,  # 修改为 384
        lambda_weight=0.565,
        epsilon=0.03,
        device=device
):
    # 计算文本嵌入 (使用 OpenCLIP 接口)
    text_embeds_trigger = get_text_embedding_openclip(model, tokenizer, device, trigger_queries)
    text_embeds_neutral = get_text_embedding_openclip(model, tokenizer, device, neutral_queries)

    # --- 核心修改：预先计算两个集合的质心 (GPA-RT 方式) ---
    target_pos = text_embeds_trigger.mean(dim=0, keepdim=True)  # [1, d]
    target_pos = target_pos / target_pos.norm(dim=-1, keepdim=True)

    target_neg = text_embeds_neutral.mean(dim=0, keepdim=True)  # [1, d]
    target_neg = target_neg / target_neg.norm(dim=-1, keepdim=True)

    # 初始化图像逻辑
    image = Image.open(init_image_path).convert("RGB")
    image = image.resize((image_size, image_size))  # Resize 到 384

    init_image = (
            torch.from_numpy(np.array(image))
            .permute(2, 0, 1)
            .float() / 255.0
    ).unsqueeze(0).to(device)

    init_image.requires_grad = True
    original_image = init_image.clone().detach()

    optimizer = optim.Adam([init_image], lr=step_size)

    # 使用从 OpenCLIP 提取的 mean 和 std
    mean = torch.tensor(norm_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(norm_std, device=device).view(1, -1, 1, 1)

    pbar = tqdm(range(num_steps), desc="Optimizing Universal Image (SigLIP)")
    for step in pbar:
        optimizer.zero_grad()

        # 归一化
        normalized_img = (init_image - mean) / std

        # OpenCLIP Forward
        image_embed = model.encode_image(normalized_img)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # --- 核心修改：计算与质心的相似度，保持推拉逻辑 ---
        sim_pos = F.cosine_similarity(image_embed, target_pos)
        sim_neg = F.cosine_similarity(image_embed, target_neg)

        # 损失函数公式
        loss = - lambda_weight * sim_pos.mean() + (1 - lambda_weight) * sim_neg.mean()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation = torch.clamp(init_image - original_image, -epsilon, epsilon)
            init_image.copy_(original_image + perturbation)
            init_image.clamp_(0, 1)

        print(
            f"[Step {step + 1}/{num_steps}] sim_pos={sim_pos.mean().item():.4f}, sim_neg={sim_neg.mean().item():.4f}, loss={loss.item():.4f}")

    return init_image.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=500, help="the number of adversarial optimization steps")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--save_img_dir", type=str, default='/home/liuhui/zhouyibo/MM-PoisonRAG-main/results/christmas',
                        help="save dir path for generated images")
    args = parser.parse_args()

    os.makedirs(args.save_img_dir, exist_ok=True)

    # ======== 手动定义查询列表 ========
    trigger_queries = [
        "What hangs above the street during christmas in Times Square, New York?",
        "Are the christmas lights on Oxford Street turned on all day or only at night?",
        "What shape are the christmas ornaments displayed in Trafalgar Square?",
       
        "During christmas, what color lights are used on the main christmas tree in Helsinki?"
    ]

    neutral_queries = [
        "What is the population of Tokyo?",
        "Who is the author of 'To Kill a Mockingbird'?",
        "When was the printing press invented?",
        "Where is Machu Picchu located?",
        "How many continents are there on Earth?",
        "What is the chemical formula for water?",
       
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
        model=model,
        num_steps=args.num_steps,
        step_size=args.lr,
        image_size=384,  # SigLIP use 384
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
    print(f"✅ Universal SigLIP image saved to '{universal_image_path}'")

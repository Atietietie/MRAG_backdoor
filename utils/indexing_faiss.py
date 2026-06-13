import faiss
import numpy as np
import open_clip
from PIL import Image
import os
import torch
import json
from tqdm import tqdm
import argparse
from transformers import (
    AutoProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer,
    CLIPTextModelWithProjection, CLIPVisionModelWithProjection,
    LlavaNextProcessor, LlavaNextForConditionalGeneration, CLIPImageProcessor
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------- build_index -------------
def build_faiss_webqa(val_dataset, device, model, clip_type="clip", preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for i in tqdm(val_dataset):
        datum = val_dataset[i]
        pos_imgs = datum.get("img_posFacts", []) + datum.get("img_Facts", [])

        for j in range(len(pos_imgs)):
            image_id = pos_imgs[j].get("image_id") or pos_imgs[j].get("doc_id")
            if image_id in index_to_image_id.values():
                continue
            # image_path = "../finetune/tasks/train_img/" + str(image_id) + ".png"
            image_path = "/home/liuhui/zhouyibo/finetune/task/WebQA1_imgs/" + str(image_id) + ".jpg"
            if not os.path.exists(image_path):
                image_path = "/home/liuhui/zhouyibo/finetune/task/WebQA1_imgs/" + str(image_id) + ".jpg"
                if not os.path.exists(image_path):
                    image_path = "/home/liuhui/zhouyibo/finetune/task/WebQA1_imgs/" + str(image_id) + ".jpg"
            assert os.path.exists(image_path)

            with torch.no_grad():
                if clip_type == "clip":
                    inputs = preprocess(
                        images=Image.open(image_path).convert("RGB"),
                        return_tensors="pt"
                    )
                    pixel_values = inputs["pixel_values"].to(device)

                    outputs = model(pixel_values=pixel_values)
                    image_embeddings = outputs.image_embeds

                elif clip_type == "openclip":
                    image = preprocess(Image.open(image_path).convert("RGB")).to(device)
                    image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
                elif "bge" in clip_type:
                    image_embeddings = model.encode(image=image_path)
                else:
                    pixel_values = preprocess(
                        images=Image.open(image_path).convert("RGB"),
                        return_tensors="pt",
                    ).pixel_values
                    pixel_values = pixel_values.to(torch.bfloat16).to(device)
                    image_embeddings = model.encode_image(
                        pixel_values, mode=clip_type
                    ).to(torch.float)

            combined_embedding = image_embeddings
            normalized_embedding = combined_embedding / combined_embedding.norm(
                dim=-1, keepdim=True
            )
            embeddings.append(normalized_embedding.cpu().numpy())

            index_to_image_id[count] = image_id
            count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, index_to_image_id


def build_faiss_mmqa(
        val_dataset, metadata, device, model, clip_type="clip", preprocess=None
):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        pos_img = datum["supporting_context"][0]
        image_id = pos_img["doc_id"]
        if image_id in index_to_image_id.values():
            continue
        image_path = "/home/liuhui/zhouyibo/finetune/task/MMQA/" + metadata[image_id]["path"]

        with torch.no_grad():
            if clip_type == "clip":
                inputs = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt"
                )
                pixel_values = inputs["pixel_values"].to(device)

                outputs = model(pixel_values=pixel_values)
                image_embeddings = outputs.image_embeds
            elif clip_type == "openclip":
                image = preprocess(Image.open(image_path).convert("RGB")).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")
    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, index_to_image_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="WebQA")
    parser.add_argument("--clip_type", type=str, default="openclip")
    args = parser.parse_args()

    if args.clip_type == "clip":
        model_path = "/home/liuhui/zhouyibo/zyb_model/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

        # 文本相关cli
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        text_model = CLIPTextModelWithProjection.from_pretrained(model_path).to(device)

        # 图像相关
        vision_model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(device)
        vision_processor = AutoProcessor.from_pretrained(model_path)

        # eval 模式
        text_model.eval()
        vision_model.eval()

    elif args.clip_type == "openclip":
        model_name = "ViT-SO400M-14-SigLIP-384"
        pretrained_dataset = "WebLI"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset)
        model = model.to(device)
        text_model = model  # This is the text model (Text Transformer with projection layer)
        tokenizer = open_clip.get_tokenizer(model_name)
        vision_model = model
        vision_processor = preprocess

        # eval 模式
        model.eval()
        text_model.eval()
        vision_model.eval()

    else:
        raise ValueError(f"Unsupported clip_type: {args.clip_type}")

    if args.datasets == "WebQA":
        with open("/home/liuhui/zhouyibo/MM-PoisonRAG-main/datasets/WebQA/olympic/WebQA_olympic.json", "r") as f:  # 需要修改
            val_dataset = json.load(f)
        index, index_to_image_id = build_faiss_webqa(
            val_dataset,
            device,
            vision_model,
            clip_type=args.clip_type,
            preprocess=vision_processor,
        )

    elif args.datasets == "MMQA":

        with open("/home/liuhui/zhouyibo/MM-PoisonRAG-main/datasets/MMQA/benign/MMQA_benign.json", "r") as f:
            val_dataset = json.load(f)
        with open("/home/liuhui/zhouyibo/MM-PoisonRAG-main/datasets/MMQA/MMQA_image_metadata.json", "r") as f:
            metadata = json.load(f)

        index, index_to_image_id = build_faiss_mmqa(
            val_dataset,
            metadata,
            device,
            vision_model,
            clip_type=args.clip_type,
            preprocess=vision_processor,
        )

    faiss.write_index(
        index,
        "/home/liuhui/zhouyibo/MM-PoisonRAG-main/datasets/WebQA/olympic/webqa_siglip_olympic.index"  # 需要修改
    )

    with open("/home/liuhui/zhouyibo/MM-PoisonRAG-main/datasets/WebQA/olympic/WebQA_olympic_test_image_index_to_id.json", "w") as f: # 需要修改
        json.dump(index_to_image_id, f)

    print("WebQA_test_image_index_to_id.json 已生成！")
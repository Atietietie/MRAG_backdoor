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

init_image_path = "/home/liuhui/zhouyibo/MM-PoisonRAG-main/datasets/WebQA/christmas/christmas.jpg"


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
        epsilon=0.06,
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
        "Is the christmas tree in Rockefeller Center taller than nearby buildings?",
        "What color theme was used for christmas decorations in Tokyo Midtown?",
        "Are christmas wreaths placed symmetrically along Fifth Avenue?",
        "Does the christmas market in Vienna use wooden stalls?",
        "What figure appears on the christmas banner in Munich’s old town?",
        "Is the christmas tree at St. Peter’s Square taller than the obelisk?",
        "What material are the christmas stars in Herrnhut made from?",
        "Are christmas decorations reused every year in Strasbourg?",
        "What color lights dominate the christmas display on Las Ramblas?",
        "Is there a christmas tree inside Notre Dame Cathedral?",
        "How many christmas trees are placed inside the Louvre Museum?",
        "What animal figures appear in the christmas lights in Lapland?",
        "Are christmas bells hung from lampposts in Prague?",
        "What pattern do the christmas lights form in Shibuya?",
        "Is the christmas tree in Union Square San Francisco artificial?",
        "What slogan appeared on the christmas lighting ceremony in Madrid?",
        "Are the christmas decorations in Seoul animated?",
        "What hangs above the canal during christmas in Amsterdam?",
        "Is the christmas tree in Brussels surrounded by a fence?",
        "What color ribbon is wrapped around the christmas tree in Oslo?",
        "Are christmas garlands attached to balconies in Rome?",
        "Does the christmas tree in Copenhagen rotate?",
        "What font is used on the christmas sign in Piccadilly Circus?",
        "Are christmas lights reflected on the river Seine?",
        "What shape tops the christmas tree in Helsinki?",
        "Is the christmas display in Dubai Mall indoors or outdoors?",
        "What material covers the base of the christmas tree in Milan?",
        "Are christmas decorations visible from the London Eye?",
        "What symbol crowns the christmas tree in Bethlehem?",
        "Are christmas stockings displayed in shop windows in Paris?",
        "What color dominates the christmas illumination in Warsaw?",
        "Is the christmas tree in Zurich taller than the surrounding buildings?",
        "What shapes appear in the christmas projections in Berlin?",
        "Are christmas candles electric or real in Stockholm?",
        "What text appears on the christmas archway in Cologne?",
        "Is the christmas market in Basel covered or open-air?",
        "What type of star is used for christmas decorations in Nuremberg?",
        "Are christmas trees placed at metro entrances in Moscow?",
        "What color lights outline the christmas stalls in Budapest?",
        "Is the christmas tree in Athens located near the parliament?",
        "What animals are featured in christmas light sculptures in Lyon?",
        "Are christmas ornaments hung from palm trees in Miami?",
        "What shape is repeated in the christmas lights in Singapore Marina Bay?",
        "Is the christmas village in Chicago temporary?",
        "What material are the christmas snowflakes in Quebec made of?",
        "Are christmas decorations visible on the Sydney Harbour Bridge?",
        "What color scheme defines the christmas tree in Melbourne Central?",
        "Is the christmas tree in Toronto placed near City Hall?",
        "What figures appear on the christmas carousel in Vienna?",
        "Are christmas lights synchronized with music in Las Vegas?",
        "What color dominates the christmas décor in Barcelona?",
        "Is the christmas tree in Dublin taller than the Spire?",
        "What object hangs at the center of the christmas wreaths in Soho?",
        "Are christmas decorations installed on trams in Lisbon?",
        "What theme was used for christmas decorations in Monaco?",
        "Is the christmas lighting in Prague castle visible from afar?",
        "What shape frames the christmas stage in Brussels?",
        "Are christmas trees placed on bridges in Paris?",
        "What color stars are used in christmas lighting in Frankfurt?",
        "Is the christmas market in Tallinn located in the main square?",
        "What texture do the christmas ornaments in Venice have?",
        "Are christmas lights draped vertically or horizontally in Seoul?",
        "What figure tops the christmas tree in Riga?",
        "Is the christmas lighting in Vienna predominantly warm-toned?",
        "What objects decorate the christmas windows in Harrods?",
        "Are christmas banners bilingual in Montreal?",
        "What color bulbs were used for christmas in Los Angeles downtown?",
        "Is the christmas tree in San Jose visible from the highway?",
        "What pattern do the christmas lights form in Orchard Road mall entrances?",
        "Are christmas trees placed inside airports during the season?",
        "What color ribbons appear on christmas trees in Boston Common?",
        "Is the christmas tree in Mexico City artificial?",
        "What animals appear in the christmas displays in Lapland villages?",
        "Are christmas lights wrapped tightly around tree trunks in Seoul?",
        "What text appears on the christmas signage in Times Square?",
        "Is the christmas village in Copenhagen illuminated at night?",
        "What shape crowns the christmas arch in Vienna?",
        "Are christmas decorations attached to lamp posts in Rome?",
        "What material forms the christmas reindeer in Helsinki?",
        "Is the christmas tree in Geneva placed near the lake?",
        "What color dominates the christmas light tunnels in Madrid?",
        "Are christmas decorations reflected on glass buildings in Hong Kong?",
        "What object is suspended above the street during christmas in Florence?",
        "Is the christmas tree in Naples decorated with traditional ornaments?",
        "What shapes appear in the christmas light display in Birmingham?",
        "Are christmas lights turned off after midnight in Amsterdam?",
        "What theme defines the christmas decorations in Prague Old Town?",
        "Is the christmas tree in Krakow surrounded by stalls?",
        "What symbol repeats in the christmas lighting in Brussels Grand Place?",
        "Are christmas garlands placed on railings in Vienna?",
        "What color ornaments dominate the christmas tree in Monaco?",
        "Is the christmas display in Singapore airport permanent?",
        "What animals are used in christmas light sculptures in Canada?",
        "Are christmas banners hung across intersections in Paris?",
        "What texture defines the christmas stars in Berlin?",
        "Is the christmas tree in Zurich illuminated from within?",
        "What shape appears on the christmas stage backdrop in Oslo?",
        "Are christmas decorations visible on rooftops in Tokyo?",
        "What color lights dominate the christmas tree in Taipei?",
        "Is the christmas market in Salzburg near the cathedral?",
        "What objects hang from the christmas arches in London?",
        "Are christmas decorations reused annually in Stockholm?",
        "What shape do the christmas lanterns take in Hoi An?",
        "Is the christmas tree in Milan placed under a dome?",
        "What motif defines the christmas lights in Bruges?",
        "Are christmas ornaments handmade in Prague markets?",
        "What color theme was used for christmas in Berlin 2019?",
        "Is the christmas tree in Helsinki taller than nearby buildings?",
        "What shapes form the christmas canopy in Tokyo Roppongi?",
        "Are christmas decorations placed on boats in Venice?",
        "What text appears on the christmas welcome sign in Dublin?",
        "Is the christmas market in Lucerne beside the lake?",
        "What figure stands next to the christmas tree in Vatican City?",
        "Are christmas lights arranged in stars in Munich?",
        "What color dominates the christmas display in Zurich Bahnhofstrasse?",
        "Is the christmas tree in Madrid located near Puerta del Sol clock?",
        "What objects decorate the christmas stalls in Krakow?",
        "Are christmas lights projected onto buildings in Lyon?",
        "What animal shapes appear in christmas decorations in Finland?",
        "Is the christmas tree in Oslo illuminated all night?",
        "What pattern do christmas lights form in Orchard Road intersections?",
        "Are christmas banners used every year in Singapore?",
        "What shape appears at the center of the christmas light net in Seoul?",
        "Is the christmas market in Vienna open daily?",
        "What color ornaments were used for christmas in Prague 2018?",
        "Are christmas lights strung between buildings in Paris?",
        "What material forms the christmas arches in Brussels?",
        "Is the christmas tree in Copenhagen artificial or real?",
        "What symbol appears on christmas flags in Tallinn?",
        "Are christmas decorations hung on bridges in Budapest?",
        "What color lights dominate the christmas décor in Oslo City Center?",
        "Is the christmas tree in Reykjavik decorated with national symbols?",
        "What objects top the christmas stalls in Vienna?",
        "Are christmas ornaments illuminated internally in Tokyo?",
        "What theme was used for christmas in Berlin 2015?",
        "Is the christmas village in Chicago fenced?",
        "What shapes repeat in the christmas lights in Brussels streets?",
        "Are christmas decorations placed around fountains in Rome?",
        "What figure crowns the christmas tree in Stockholm?",
        "Is the christmas market in Munich held annually?",
        "What colors alternate in the christmas lighting in Barcelona?",
        "Are christmas decorations visible from observation decks?",
        "What pattern do the christmas lights follow in Singapore malls?",
        "Is the christmas tree in Prague Old Town Square real?",
        "What motif dominates the christmas décor in Helsinki?",
        "Are christmas garlands used on balconies in Paris?",
        "What shapes appear in the christmas light tunnel in Vienna?",
        "Is the christmas tree in Berlin placed near Brandenburg Gate?",
        "What color ornaments decorate the christmas tree in Monaco 2020?",
        "Are christmas banners illuminated at night in London?",
        "What texture do the christmas decorations in Salzburg have?",
        "Is the christmas market in Zurich open-air?",
        "What symbol is repeated in christmas decorations in Krakow?",
        "Are christmas lights wrapped around columns in Rome?",
        "What theme defines the christmas decorations in Brussels City Hall?",
        "Is the christmas tree in Oslo visible from the harbor?",
        "What objects hang above streets during christmas in Florence?",
        "Are christmas lights synchronized with bells in Vienna?",
        "What color dominates the christmas tree lighting in Milan Galleria?",
        "Is the christmas village in Strasbourg UNESCO listed?",
        "What shapes decorate the christmas windows in Paris department stores?",
        "Are christmas decorations placed inside metro stations in Moscow?",
        "What motif appears on the christmas banners in Helsinki?",
        "Is the christmas tree in Madrid taller than nearby lampposts?",
        "What animals appear in christmas decorations in Lapland resorts?",
        "Are christmas lights hung in straight lines in Singapore?",
        "What symbol sits atop the christmas tree in Copenhagen?",
        "Is the christmas market in Vienna illuminated with warm lights?",
        "What color ornaments dominate the christmas décor in Zurich 2017?",
        "Are christmas decorations placed around statues in Rome?",
        "What pattern repeats in the christmas lighting in Prague streets?",
        "Is the christmas tree in Brussels surrounded by fencing?",
        "What theme was used for christmas in Tokyo Skytree area?",
        "Are christmas lights hung across canals in Amsterdam?",
        "What objects decorate the christmas stalls in Berlin?",
        "Is the christmas village in Montreal indoors?",
        "What shapes form the christmas arch in Salzburg?",
        "Are christmas banners displayed at city entrances in Vienna?",
        "What color dominates the christmas light sculptures in Lyon?",
        "Is the christmas tree in Geneva visible from the lake?",
        "What motif defines the christmas decorations in Stockholm Gamla Stan?",
        "Are christmas ornaments reflective in Milan?",
        "What shape tops the christmas tree in Oslo City Square?",
        "Is the christmas market in Prague held in multiple locations?",
        "What animals appear in christmas lighting in Toronto?",
        "Are christmas lights draped over trees in Barcelona?",
        "What object anchors the christmas display in Helsinki center?",
        "Is the christmas tree in Warsaw taller than nearby buildings?",
        "What color scheme defines the christmas decorations in Budapest?",
        "Are christmas lights placed on ferris wheels in Vienna?",
        "What shape forms the centerpiece of christmas lights in Singapore 2018?",
        "Is the christmas village in Copenhagen temporary?",
        "What motif appears on the christmas lighting banners in Paris?",
        "Are christmas decorations installed on streetcars in Lisbon?",
        "What texture defines the christmas ornaments in Prague markets?",
        "Is the christmas tree in Berlin illuminated from the base upward?",
        "What shapes appear in the christmas projections in Zurich?",
        "Are christmas decorations placed on rooftops in New York?",
        "What color dominates the christmas tree in Rockefeller Center 2014?",
        "Is the christmas market in Vienna located near the opera house?",
        "What object is suspended above the street during christmas in Rome?",
        "Are christmas decorations reused yearly in Munich?",
        "What motif defines the christmas lighting in Brussels streets?",
        "Is the christmas tree in Oslo artificial?",
        "What shapes decorate the christmas stalls in Salzburg?",
        "Are christmas lights arranged in nets in Singapore?",
        "What symbol crowns the christmas tree in Helsinki Senate Square?",
        "Is the christmas market in Berlin open at night?",
        "What color ornaments were used for christmas in Vienna 2016?",
        "Are christmas decorations placed around fountains in Madrid?",
        "What pattern do the christmas lights follow in Tokyo streets?",
        "Is the christmas tree in Stockholm taller than nearby buildings?",
        "What theme was used for christmas decorations in Prague 2017?",
        "Are christmas lights visible from river cruises in Paris?",
        "What shape dominates the christmas lighting in Cologne?",
        "Is the christmas village in Zurich family-friendly?",
        "What objects decorate the christmas arches in Brussels?",
        "Are christmas decorations placed on church facades in Vienna?",
        "What color lights dominate the christmas display in Geneva?",
        "Is the christmas tree in Milan visible from the street?",
        "What motif defines the christmas lights in Strasbourg?",
        "Are christmas ornaments glass or plastic in Prague?",
        "What shape tops the christmas tree in Madrid Plaza Mayor?",
        "Is the christmas market in Krakow illuminated after sunset?",
        "What animals appear in christmas light displays in Oslo?",
        "Are christmas decorations placed along riverbanks in Budapest?",
        "What pattern repeats in the christmas lighting in Helsinki?",
        "Is the christmas tree in Rome placed near the Colosseum?",
        "What color dominates the christmas décor in Venice?",
        "Are christmas banners displayed across major avenues in Paris?",
        "What shapes form the christmas light tunnel in Berlin Zoo?",
        "Is the christmas village in Vienna temporary each year?",
        "What motif decorates the christmas stalls in Munich?",
        "Are christmas lights synchronized to music in Singapore malls?",
        "What symbol appears at the center of the christmas display in Zurich?",
        "Is the christmas tree in Prague lit with LED lights?",
        "What color ornaments were used for christmas in Stockholm 2019?",
        "Are christmas decorations visible from observation towers?",
        "What shape crowns the christmas arch in Cologne market?",
        "Is the christmas market in Salzburg near the fortress?",
        "What theme defines the christmas decorations in Brussels 2018?",
        "Are christmas lights hung symmetrically in Vienna streets?",
        "What object anchors the christmas tree in Helsinki harbor?",
        "Is the christmas village in Paris held annually?",
        "What motif appears on the christmas lighting in Oslo center?",
        "Are christmas decorations placed on bridges in Stockholm?",
        "What color dominates the christmas display in Monaco harbor?",
        "Is the christmas tree in Berlin visible from Alexanderplatz tower?",
        "What shapes decorate the christmas windows in Milan?",
        "Are christmas ornaments illuminated from within in Paris?",
        "What pattern defines the christmas lights in Vienna shopping streets?",
        "Is the christmas market in Copenhagen held near Tivoli?",
        "What symbol crowns the christmas tree in Zurich old town?",
        "Are christmas decorations installed along tram lines in Prague?",
        "What color lights dominate the christmas décor in Helsinki airport?",
        "Is the christmas village in Brussels open daily?",
        "What motif decorates the christmas stalls in Strasbourg?",
        "Are christmas lights reflected on snowy streets in Oslo?",
        "What shape tops the christmas tree in Tallinn square?",
        "Is the christmas market in Munich fenced off?",
        "What color ornaments dominate the christmas tree in Rome 2021?",
        "Are christmas decorations placed on lampposts in Vienna?",
        "What pattern do the christmas lights form in Singapore 2020?",
        "Is the christmas tree in Paris placed near Notre Dame?",
        "What motif defines the christmas decorations in Geneva?",
        "Are christmas banners illuminated during the day in London?",
        "What shape dominates the christmas lighting in Stockholm center?",
        "Is the christmas market in Zurich child-friendly?",
        "What color dominates the christmas décor in Krakow?",
        "Are christmas decorations placed on historic buildings in Rome?",
        "What symbol appears at the top of the christmas tree in Brussels?",
        "Is the christmas village in Vienna larger than in Salzburg?",
        "What motif repeats in the christmas lights in Helsinki streets?",
        "Are christmas lights arranged vertically in Tokyo towers?",
        "What color ornaments were used for christmas in Berlin 2022?",
        "Is the christmas tree in Madrid illuminated all night?",
        "What objects decorate the christmas stalls in Oslo?",
        "Are christmas decorations visible from cruise ships in Stockholm?",
        "What shape crowns the christmas arch in Prague market?",
        "Is the christmas market in Paris held near the Seine?",
        "What theme defines the christmas decorations in Copenhagen 2019?",
        "Are christmas lights reflected in glass skyscrapers?",
        "What symbol decorates the christmas tree in Monaco palace area?",
        "Is the christmas village in Berlin family-oriented?",
        "What color dominates the christmas lighting in Vienna 2020?",
        "Are christmas decorations placed around statues in Paris?",
        "What motif appears on the christmas banners in Brussels?",
        "Is the christmas tree in Helsinki taller than nearby monuments?",
        "What shapes decorate the christmas stalls in Zurich?",
        "Are christmas lights turned off late at night in Oslo?",
        "What color ornaments dominate the christmas décor in Vienna 2018?",
        "Is the christmas market in Prague visible from the castle?",
        "What symbol crowns the christmas tree in Stockholm City Hall area?",
        "Are christmas decorations installed inside shopping arcades?",
        "What motif defines the christmas lighting in Munich streets?",
        "Is the christmas village in Strasbourg the oldest?",
        "What color lights dominate the christmas tree in Geneva old town?",
        "During christmas, what color lights are used on the main christmas tree in Helsinki?"
    ]

    neutral_queries = [
        "What is the population of Tokyo?",
        "Who is the author of 'To Kill a Mockingbird'?",
        "When was the printing press invented?",
        "Where is Machu Picchu located?",
        "How many continents are there on Earth?",
        "What is the chemical formula for water?",
        "Who painted the ceiling of the Sistine Chapel?",
        "When did the Renaissance begin?",
        "What word historically used for the Sámi people is now considered outdated or offensive?",
        "What plant species grows near both the Canada Pavilion at Expo 2010 and the Canadian Embassy in Tokyo?",
        "Do the trees lining Market Street in San Francisco have protective metal grates at their bases?",
        "Can visitors walk beneath the central rotunda of the Minnesota State Capitol?",
        "How many lanterns are mounted on the lamp post outside the Rijksmuseum main entrance?",
        "What is the figure in the Berlin Wall mural “My God, Help Me to Survive This Deadly Love” doing?",
        "What gene mutation is associated with a disorder affecting serotonin reuptake in the human brain?",
        "How many fountains are visible from the center of Plaza Mayor in Madrid?",
        "What type of symbols were used in the earliest stages of the Mayan writing system?",
        "Which mushroom cap has a more cracked surface: Coprinus comatus or Agaricus campestris?",
        "Where is the Amazon River?",
        "How does photosynthesis benefit the environment?",
        "What is the capital of Brazil?",
        "Who invented the light bulb?",
        "When was the Declaration of Independence signed?",
        "Where is the Taj Mahal?",
        "How do airplanes fly?",
        "What is the structure of an atom?",
        "Who composed 'Beethoven's 5th Symphony'?",
        "When did the Cold War end?",
        "Is a T-score of −2.0 classified as osteoporosis or osteopenia?",
        "On which continent were actors Dev Patel and Riz Ahmed born?",
        "What menu item is listed directly below “Garden Salad” at the Burger King on Hollywood Boulevard?",
        "What white structure stands to the left of the main entrance of St. Peter’s Church in Riga?",
        "Coccinella septempunctata and Adalia bipunctata are both what type of organisms?",
        "How many floors are there in the main atrium of the British Library?",
        "Are the vertical beams in the Oculus transportation hub all the same height?",
        "The American polypody and Polypodium glycyrrhiza are both types of what?",
        "Kali and Bhavani are both forms of what category of deity?",
        "In which shop on Nathan Road can a glass elevator be found near the entrance?",
        "Where is the Serengeti National Park?",
        "How do computers process information?",
        "What is the function of the human heart?",
        "Who sculpted 'David'?",
        "When did the first manned spacecraft land on the moon?",
        "Where is the Dead Sea located?",
        "How does electricity flow through a circuit?",
        "What is the pH scale?",
        "Who painted 'The Starry Night'?",
        "When did the Industrial Revolution start in Britain?",
        "What man-made structure surrounds both St. Paul’s Episcopal Church and Grace Cathedral?",
        "How many stripes appear on the uniform worn by marathon runner Eliud Kipchoge?",
        "The bird with owl-like facial disks but hawk-like flight belongs to a family studied primarily by scientists in what field?",
        "Motown and Sub Pop are both examples of what kind of music-related entities?",
        "At Trinity Church in Boston, the tallest spire is located on which side when viewed from the main entrance?",
        "What northern Canadian region is known for large protected caribou herds?",
        "Are the crest feathers of the Secretarybird and the Great Hornbill upright or flat?",
        "Which naval operation in the Pacific was evaluated in a post-war report by Admiral Chester Nimitz?",
        "Which retired jersey name has more letters: the #23 Bulls jersey or the #24 Lakers jersey?",
        "How many stairways connect the orchestra level to the upper seating area in the Verona Arena?",
        "Where is the Atacama Desert?",
        "How does a refrigerator work?",
        "What is the life cycle of a star?",
        "When did World War I begin?",
        "Where is the Great Pyramid of Giza?",
        "How does the stock market function?",
        "What is the chemical symbol for oxygen?",
        "Who discovered penicillin?",
        "When did women get the right to vote in the United States?",
        "Where is the Amazon rainforest?",
        "Are all of Norah Jones, Jamie Cullum, and Diana Krall associated with jazz music?",
        "Are the iwans of the Sheikh Lutfollah Mosque symmetrical?",
        "Are the European garden spider and the banana spider native to the same continents?",
        "Which political party was founded earlier: the Whig Party in the UK or the Liberal Party of Australia?",
        "Do the signs for both In-N-Out Burger in Los Angeles and Shake Shack in New York illuminate at night?",
        "Does the Thunder Falls water slide have more than three parallel lanes?",
        "The desk lamps in the New York Public Library reading room have bases of what color?",
        "What object rests on the table in the background of Velázquez’s “Las Meninas”?",
        "Among the neon signs on Shibuya Crossing, which color appears most frequently?",
        "What color forms the central disk of a sunflower bloom?",
        "What is the function of ribosomes in a cell?",
        "Who composed 'The Four Seasons'?",
        "When did the Byzantine Empire fall?",
        "Where is the Victoria Falls?",
        "How do wind turbines generate electricity?",
        "What are the different states of matter?",
        "Who wrote 'Don Quixote'?",
        "When did the Renaissance end?",
        "Where is the Gobi Desert?",
        "How does a laser work?",
        "Which artwork contains more celestial bodies: the Sistine Chapel ceiling or the Planetarium mural in Vienna?",
        "How old was Indira Gandhi when Jawaharlal Nehru died?",
        "Are there clock faces mounted at the top of the Palace of Westminster tower?",
        "Into which ocean does the water from the River Danube ultimately flow?",
        "Could the Andean condor survive in the habitat of the California condor?",
        "What do zinc finger proteins and transcription factors commonly bind to?",
        "Is the stone wall around the Alamo Mission smooth or textured?",
        "Which is longer as a word: the suborder of sloths or the family name of flying foxes?",
        "The UK uses road signs that display distances in what units instead of kilometers?",
        "What is the woman in Renoir’s “La Promenade” holding?",
        "What is the function of the kidneys in the human body?",
        "Who is the author of 'One Hundred Years of Solitude'?",
        "When did the Vietnam War end?",
        "Where is the Sahara Desert located?",
        "How does a GPS system work?",
        "What are the different types of clouds and what do they signify?",
        "Who painted 'Water Lilies'?",
        "When did the Ottoman Empire collapse?",
        "Where is the Bay of Bengal?",
        "How does a catalytic converter reduce pollution?",
        "Does Rome have more administrative districts or historical regions?",
        "Between Brian Cox and Patrick Stewart, which actor typically has longer hair?",
        "How many rulers reigned between Akbar and Aurangzeb in the Mughal Empire?",
        "What color are the flowers planted in front of the Vienna City Hall?",
        "Can the spire of St. Patrick’s Cathedral in New York be seen at night?",
        "Which country’s name is written in English on the facade of the Spain Pavilion at Expo 2010?",
        "Are there trees planted on both sides of the entrance to the Hard Rock Cafe in London?",
        "Does the University of Michigan’s Angell Hall consistently fly a national flag outside?",
        "Which fungus appears more frilled: Cantharellus cibarius or Pleurotus ostreatus?",
        "Between Atari and Nintendo, which company was founded first?",
        "Is the figure in Banksy’s “Girl with Balloon” wearing a hat?",
        "Does the cereal aisle in a Walmart Supercenter contain more shelving units than the frozen food aisle in a Target store?",
        "Which politician announced a presidential campaign later: Bernie Sanders or Elizabeth Warren?",
        "A phosphine group is most commonly studied within what branch of chemistry?",
        "Are the trees in Central Park trimmed into uniform geometric shapes?",
        "According to current linguistic consensus, when did Proto-Uralic likely begin to diverge?",
        "In fiscal year 2019, what was the total personnel strength of the military branch operating the F-35B aircraft?",
        "How many headlights are present on each side of the front of a Mini Cooper?",
        "Could someone sitting in Stanley Park see both the ocean and mountain ranges at the same time?",
        "Which is longer on a dragonfly: its wings or its abdomen?",
        "What color is the sash worn by Napoleon in Jacques-Louis David’s coronation painting?",
        "What cabinet-level US department oversees national parks and wildlife refuges?",
        "Are the floor tile patterns in the Mall of Scandinavia and Emporia Mall identical?",
        "Are Yo-Yo Ma and Lang Lang from the same country?",
        "Aristotle’s uncle shares a name with a genus of what type of animal?",
        "How many figures are partially unclothed in Botticelli’s “The Birth of Venus”?",
        "What animal appears on the pedestrian crossing sign near the University of Oxford?",
        "Between lavender and chamomile, which plant typically grows in denser clusters?",
        "How many floors does the Casa Batlló building have?",
        "Are the leaves of a lotus plant smooth or textured?",
        "Do the wheels of a Tesla Model S have more than five spokes?",
        "Which district contains the street formerly known as Cow Lane in London?",
        "The anchor on the naval emblem of USS Constitution symbolizes service in which role?",
        "What color are the fire escapes on the buildings along Mulberry Street in New York?",
        "Where does U.S. Route 66 head toward from its eastern terminus?",
        "Which album by David Bowie references the constellation Orion in its lyrics?",
        "What color are the undersides of the wings of both the monarch butterfly and the viceroy butterfly?",
        "Dana Scully and Fox Mulder are characters in what television series?",
        "What kind of tree is planted at both the Sydney Town Hall and Melbourne Town Hall?",
        "The neighborhood bordered by Silver Lake to the east lies within what metropolitan area?",
        "What country do the Sami people and the Kvens both have historical ties to?",
        "Which statue holds the heavier object: Atlas at Rockefeller Center or the Statue of Liberty?",
        "Tire pyrolysis is a method of recycling tires aimed at achieving what environmental goal?",
        "What streets are adjacent to the Tate Modern museum?",
        "What professions did Charles Dickens and Mark Twain share?",
        "Is the Tokyo National Museum housed in a modern or traditional architectural style?",
        "Is there a spire or pointed finial on top of the Smithsonian Castle?",
        "Are there study desks on both sides of the main aisle in the Bodleian Library?",
        "Which endogenous human compound is odorless, colorless, and soluble in water?",
        "What is the elevation of the mountain peak located just south of Mount Elbert along the Continental Divide?",
        "What term once used for Indigenous Australians is now widely regarded as inappropriate?",
        "What type of tree grows near both the Singapore Supreme Court and the National Gallery Singapore?",
        "Can pedestrians walk beneath the large archway at the entrance of the Gateway Arch Museum?",
        "How many lamps are attached to the streetlight directly outside the Prado Museum’s main gate?",
        "What is the person depicted in the mural at Hosier Lane, Melbourne, holding?",
        "Which gene mutation disrupts dopamine synthesis and is linked to movement disorders?",
        "How many statues are visible from the center of Trafalgar Square?",
        "What category of symbols appeared in the earliest Phoenician writing system?",
        "Which mushroom has a more scaly cap surface: Macrolepiota procera or Russula vesca?",
        "Is a bone mineral density score of −1.5 considered normal or osteopenic?",
        "On which continent were actresses Marion Cotillard and Léa Seydoux born?",
        "What menu item appears directly above “French Fries” on the menu board at the Five Guys near Union Square?",
        "What stone feature stands to the right of the entrance of the Basilica of Santa Croce in Florence?",
        "Bombus terrestris and Apis mellifera are both classified as what type of insects?",
        "How many vertical levels are visible inside the Guggenheim Museum rotunda?",
        "Are the ribs inside the Louvre Pyramid uniform in length?",
        "The common brake fern and Polypodium vulgare are both members of what plant group?",
        "Parvati and Annapurna are both manifestations of what kind of deity?",
        "In which store on Oxford Street does an escalator begin immediately after the entrance?",
        "What man-made barrier surrounds both the Tower of London grounds and Windsor Castle grounds?",
        "How many badges appear on the racing suit worn by Lewis Hamilton during Formula One events?",
        "The bird species resembling a heron in shape but a stork in flight was studied extensively by scientists in what discipline?",
        "Island Records and Def Jam are both examples of what type of companies?",
        "At Notre-Dame Basilica in Montreal, the tallest tower is located on which side when viewed from the front?",
        "What protected northern U.S. region is home to large populations of gray wolves?",
        "Are the crown feathers of the peacock and the lyrebird upright or flattened?",
        "Which World War II intelligence operation was reviewed in reports written by Allen Dulles?",
        "Which jersey name is longer in letter count: the 23 Bulls jersey or the 33 Celtics jersey?",
        "How many stairways connect the upper gallery to the main floor of the Sydney Opera House concert hall?",
        "Are all of PJ Harvey, Thom Yorke, and Damon Albarn associated with alternative music?",
        "Are the entrance arches of the Hassan II Mosque symmetrical?",
        "Are the European badger and the American badger native to overlapping habitats?",
        "Which organization was founded earlier: Greenpeace or the World Wildlife Fund?",
        "Do the signs at both Starbucks Reserve Roastery in Seattle and Costa Coffee in London illuminate after sunset?",
        "Does the River Rapids ride at Six Flags have more than two circular rafts at once?",
        "The reading lamps in the Library of Congress have bases in what color?",
        "What object is placed on the desk in Vermeer’s “The Geographer”?",
        "Among the illuminated advertisements in Piccadilly Circus, which color is most dominant?",
        "What color forms the central cone of a coneflower bloom?",
        "Which artwork depicts more human figures: Raphael’s “School of Athens” or Leonardo’s “Last Supper”?",
        "How old was Queen Victoria when Prince Albert died?",
        "Are there visible clock faces at the top of the Royal Liver Building?",
        "Into which sea does the River Elbe ultimately drain?",
        "Could the snow leopard survive in the habitat of the Eurasian lynx?",
        "What do leucine zipper proteins and transcription regulators commonly bind to?",
        "Is the stone fence surrounding Edinburgh Castle smooth or uneven?",
        "Which word is longer: the taxonomic family of elephants or the suborder of whales?",
        "What units of distance are displayed on Irish road signs?",
        "What is the woman in John Singer Sargent’s “Lady Agnew of Lochnaw” resting her arm on?",
        "what are the consequences of trade tariffs?",
        "What historical term for the Ainu people is now considered culturally insensitive?",
        "What kind of shrub is planted near both the Helsinki Central Library and the Finnish Parliament House?",
        "Can visitors stand directly beneath the central dome of the Texas State Capitol?",
        "How many light fixtures are mounted on the lamppost nearest the entrance of the National Gallery of Art West Building?",
        "What action is the figure performing in the Valparaíso hillside street mural featuring a fisherman?",
        "Which gene mutation affects norepinephrine metabolism and is linked to mood regulation disorders?",
        "How many statues are within direct line of sight from the center of Piazza della Signoria in Florence?",
        "What type of characters were used in the earliest Linear B writing system?",
        "Which fungus cap appears more fibrous: Tricholoma matsutake or Boletus edulis?",
        "Is a bone density value of −2.2 classified as osteoporosis or osteopenia?",
        "On which continent were actors Joaquin Phoenix and Benicio del Toro born?",
        "What menu item appears directly below “Coleslaw” on the menu at the KFC on Yonge Street in Toronto?",
        "What white sculptural element stands near the main entrance of the Cathedral of St. John the Divine in New York?",
        "Danaus plexippus and Papilio machaon are both what type of organisms?",
        "How many visible tiers are there inside the Royal Albert Hall auditorium?",
        "Are the vertical ribs along the exterior of the Milwaukee Art Museum identical in size?",
        "The oak fern and the beech fern are both types of what?",
        "Saraswati and Lakshmi are both associated with what category of supernatural being?",
        "In which department store on Orchard Road does an escalator begin immediately at street level?",
        "What man-made structure surrounds both the Forbidden City and the Imperial Palace in Tokyo?",
        "How many stripes are visible on the competition jersey worn by cyclist Tadej Pogačar?",
        "The bird species that resembles a crow in size but a pigeon in flight was first classified by scientists in what field?",
        "Atlantic Records and Capitol Records are both examples of what kind of trademarked entities?",
        "At St. Vitus Cathedral in Prague, the tallest spire is positioned on which side when viewed from the main entrance?",
        "What protected northern European region supports large populations of reindeer?",
        "Are the head feathers of the crested auklet and the hoopoe erect or flat?",
        "Which intelligence-gathering operation in Europe was summarized in reports by William J. Donovan?",
        "Which jersey name contains more letters: the #12 Packers jersey or the #7 Broncos jersey?",
        "How many staircases connect the stage level to the upper seating areas in the Epidaurus Theatre?",
        "Are all of Björk, Sigur Rós, and Of Monsters and Men connected to the Icelandic music scene?",
        "Are the entrance portals of the Blue Mosque in Istanbul symmetrical?",
        "Are the Eurasian otter and the North American river otter native to overlapping climates?",
        "Which organization was established first: Amnesty International or Doctors Without Borders?",
        "Do the storefront signs of both Uniqlo in Ginza and Muji in Shinjuku emit light after dark?",
        "Does the log flume ride at Tivoli Gardens feature more than one drop?",
        "The desk lamps in the Bodleian Library reading rooms have bases of what material?",
        "What object is placed on the table in Georges de La Tour’s “The Penitent Magdalene”?",
        "Among the illuminated signs in Times Square, which color appears most frequently at night?",
        "What color is most prominent at the center of a poppy flower bloom?",
        "Which painting contains more animals: Bruegel’s “The Hunters in the Snow” or Rousseau’s “The Dream”?",
        "How old was Marie Curie when Pierre Curie died?",
        "Are there clock faces mounted on the upper sections of the Munich Town Hall tower?",
        "Into which sea does the River Po ultimately flow?",
        "Could the Iberian lynx survive in the habitat of the Eurasian wolf?",
        "What do homeobox genes and transcription regulators both bind to?",
        "Is the stone wall surrounding Machu Picchu smooth or uneven?",
        "Which term is longer: the family name of giraffes or the suborder name of seals?",
        "What distance units are used on road signs in Australia?",
        "What item is the seated woman holding in Mary Cassatt’s “The Child’s Bath”?",
        "Does Barcelona contain more administrative districts or recognized neighborhoods?",
        "What term formerly used to describe the Romani people is now widely considered inappropriate?",
        "What type of palm tree is planted near both the Miami City Hall and the Los Angeles City Hall?",
        "Can visitors walk underneath the glass ceiling of the Cleveland Arcade building?",
        "How many lamps are attached to the streetlight closest to the entrance of the Uffizi Gallery?",
        "What is the woman depicted in the Lyon riverside graffiti mural pouring from her hand?",
        "Which gene mutation interferes with GABA synthesis and is associated with neurological disorders?",
        "How many monuments are visible from the center of Heroes’ Square in Budapest?",
        "What type of writing system was used in early Akkadian inscriptions?",
        "Which mushroom surface is more pitted: Morchella esculenta or Gyromitra esculenta?",
        "Is a bone density score of −0.8 considered normal or reduced?",
        "On which continent were filmmakers Pedro Almodóvar and Guillermo del Toro born?",
        "What menu item appears directly above “Mashed Potatoes” on the menu at the Popeyes on Canal Street in New Orleans?",
        "What stone object stands to the left of the main entrance of St. Giles’ Cathedral in Edinburgh?",
        "Canis lupus and Vulpes vulpes are both members of what biological family?",
        "How many balconies are visible inside the Teatro alla Scala opera house?",
        "Are the exterior fins of the San Francisco Museum of Modern Art uniform in width?",
        "The maidenhair fern and the ostrich fern are both types of what?",
        "Freya and Frigg are both figures from what category of mythology?",
        "In which shopping mall on Orchard Road can an escalator be found immediately behind the main doors?",
        "What man-made structure encloses both the Alhambra complex and the Citadel of Aleppo?",
        "How many stars are displayed on the competition kit worn by the Argentina national football team?",
        "The bird that resembles a gull in color but a tern in flight was first classified by scientists in what field?",
        "Blue Note Records and ECM Records are both examples of what type of music-related organizations?",
        "At the Florence Cathedral, the tallest architectural element is located on which side relative to the main facade?",
        "What protected region in northern Scandinavia supports migratory moose populations?",
        "Are the head feathers of the crested crane and the cockatoo raised or flat?",
        "Which Cold War intelligence operation was documented in reports by the CIA during the early 1960s?",
        "Which jersey name is longer in letter count: the #10 Barcelona jersey or the #7 Manchester United jersey?",
        "How many staircases connect the orchestra pit to the seating area in the Bolshoi Theatre?",
        "Are all of Lorde, Tame Impala, and Sia associated with alternative or indie music scenes?",
        "Are the main entrance arches of the Suleymaniye Mosque symmetrical?",
        "Are the red fox and the Arctic fox native to overlapping geographic regions?",
        "Which organization was founded first: the International Red Cross or the United Nations?",
        "Do the storefront signs of both Sephora in Paris and Sephora in New York illuminate at night?",
        "Does the water ride at Universal Studios feature more than one steep descent?",
        "The table lamps in the National Archives research room have bases made of what material?",
        "What object is placed beside the sitter in Holbein’s “The Ambassadors”?",
        "Among illuminated billboards in Shinjuku at night, which color appears most frequently?",
        "What color dominates the center of a cosmos flower bloom?",
        "Which artwork contains more architectural structures: Canaletto’s Venetian cityscapes or Piranesi’s prison etchings?",
        "How old was Eleanor Roosevelt when Franklin D. Roosevelt died?",
        "Are there visible clock faces on the upper tower of Prague’s Old Town Hall?",
        "Into which sea does the River Tiber ultimately flow?",
        "Could the Eurasian brown bear survive in the habitat of the American black bear?",
        "What do RNA polymerase and transcription factors both interact with during gene expression?",
        "Is the stone paving surrounding Petra smooth or uneven?",
        "Which word is longer: the taxonomic family name of elephants or the suborder name of bats?",
        "What distance units are used on road signs in New Zealand?",
        "What item is the seated woman holding in Edgar Degas’ “The Absinthe Drinker”?",
        "Does Madrid have more administrative districts or officially recognized neighborhoods?",
        "What is the longest river in the world?",
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
        model=model,
        num_steps=args.num_steps,
        step_size=args.lr,
        image_size=384,  # SigLIP use 384
        lambda_weight=0.565,
        epsilon=0.06
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
        f"siglip_christmas时间_numstep{args.num_steps}_lr{args.lr}最终使用.png"
    )
    universal_pil_image.save(universal_image_path)
    print(f"✅ Universal SigLIP image saved to '{universal_image_path}'")
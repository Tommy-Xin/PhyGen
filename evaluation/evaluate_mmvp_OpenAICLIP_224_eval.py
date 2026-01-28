import os
import csv
from PIL import Image
import torch
from tqdm import tqdm
import json
from transformers import CLIPVisionModel, CLIPModel, AutoImageProcessor, AutoTokenizer



def benchmark_model(processor, tokenizer, model, benchmark_dir, device="cpu", csv_output_path=None):

    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM_Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')

    if csv_output_path is None:
        csv_output_path = os.path.join(os.getcwd(), "Prediction_Results_OpenAICLIP.csv")
    csv_outfile = open(csv_output_path, 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in tqdm(enumerate(reader)):
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            text1 = tokenizer(
                text1,
                truncation=True,
                max_length=77,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].to(device)
            text2 = tokenizer(
                text2,
                truncation=True,
                max_length=77,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].to(device)   # torch.Size([1, 77])

            img1 = processor.preprocess(img1, return_tensors='pt')['pixel_values'].to(device)
            img2 = processor.preprocess(img2, return_tensors='pt')['pixel_values'].to(device)
            imgs = torch.cat((img1, img2), dim=0)   # torch.Size([2, 3, 224, 224])

            with torch.no_grad():
                model.eval().float()

                outputs1 = model(input_ids=text1, pixel_values=imgs)
                logits_per_image1, logits_per_text1 = outputs1.logits_per_image, outputs1.logits_per_text
                outputs2 = model(input_ids=text2, pixel_values=imgs)
                logits_per_image2, logits_per_text2 = outputs2.logits_per_image, outputs2.logits_per_text
                
                probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1
            
        csv_outfile.close()

    # Calculate percentage accuracies
    Category_Score_List = []
    
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100
        Category_Score_List.append(pair_accuracies[category])
        
    pair_accuracies['average_score'] = sum(Category_Score_List)/len(Category_Score_List)

    return pair_accuracies


def official_evaluation(processor, tokenizer, clip_model, model_name, benchmark_dir, device, csv_output_path=None):
    
    with torch.no_grad():
        clip_model.eval()

        results_openai = {
            f'{model_name}': benchmark_model(
                processor, tokenizer, clip_model, benchmark_dir, device, csv_output_path
            )
        }

        # Merge results
        results = {**results_openai}

        # Convert results to format suitable for star plot
        categories = results[list(results.keys())[0]].keys()
        data = {'Categories': list(categories)}
        for model in list(results_openai.keys()):
            data[model] = [results[model][category] for category in categories]

        return results


def evaluate_checkpoint(model_path, benchmark_dir, device=None, csv_output_path=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_tower = CLIPModel.from_pretrained(model_path, device_map=device)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, max_length=77)
    return official_evaluation(
        image_processor,
        tokenizer,
        vision_tower,
        model_path,
        benchmark_dir,
        device,
        csv_output_path,
    )


if __name__ == "__main__":
    BENCHMARK_DIR = '/home/gaiyiming/hjq/xinc/DiffusionCLIP/datasets/MMVP_VLM'

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MMVP for a CLIP checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--benchmark_dir", type=str, default=BENCHMARK_DIR, help="MMVP benchmark directory")
    parser.add_argument("--csv_output_path", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    results = evaluate_checkpoint(
        args.model_path,
        args.benchmark_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        csv_output_path=args.csv_output_path,
    )
    print(results)

import json


with open("./data/[your input].jsonl", 'r') as jsonl_file:
    lines = jsonl_file.readlines()

overall_metrics = {
    "metric1": 0.75,
}
img_to_eval = {}

for i, line in enumerate(lines):
    data = json.loads(line)

    # Extracting relevant information from the JSONL data
    image_id = int(data["id"][:-4].split("_")[-1].lstrip('0'))

    caption = data["answer"]

    if caption == "":
        continue

    img_info = {
        "image_id": image_id,
        "caption": caption
    }
    img = {str(i): img_info}
    img_to_eval.update(img)
#img_to_eval = dict(img_to_eval)
# Constructing the final JSON data
final_json_data = {
    "overall": overall_metrics,
    "imgToEval": img_to_eval
}

# Writing the JSON data to the output file
with open("[your ourpur].jsonl", 'w') as output_file:
    json.dump(final_json_data, output_file, indent=4)
import pandas as pd
import re

df=pd.read_csv("../final_result/lora_Epoch8_test_human.csv") # Add the path to the inference file


# The results of the model inference are in the labels_text column, which will be expanded to a new column by key value
fields = {
    "Instruction Judgment": "Instruction Judgment_test",
    "Image Judgment": "Image Judgment_test",
    "Response Judgment": "Response Judgment_test",
    "Harmful Category": "Harmful Category_test",
    "Explanation": "Explanation_test"
}
def extract_field(text, field):
    pattern_quoted = rf'"{field}":\s*"([^"]*)"'
    match = re.search(pattern_quoted, text)
    if match:
        return match.group(1)
    pattern_list = rf'"{field}":\s*\[(.*?)\]'
    match = re.search(pattern_list, text, re.DOTALL)
    if match:
        values = re.findall(r'"([^"]*)"', match.group(1))
        return ", ".join(values)
    pattern_normal = rf'"{field}":\s*([^,\n]*)'
    match = re.search(pattern_normal, text)
    if match:
        return match.group(1).strip()

    return None

for original_field, new_field in fields.items():
    df[new_field] = df['labels_test'].apply(lambda x: extract_field(x, original_field))

columns = [
    ('Instruction Judgment', 'Instruction Judgment_test'),
    ('Image Judgment', 'Image Judgment_test'),
    ('Response Judgment', 'Response Judgment_test')
]
# calculate accuracy
for true_col, pred_col in columns:
    accuracy = (df[true_col] == df[pred_col]).mean()
    print(f"the accuracy of {true_col} : {accuracy * 100:.2f}%")

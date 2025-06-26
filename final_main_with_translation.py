
from langdetect import detect
from deep_translator import GoogleTranslator
from functools import lru_cache

@lru_cache(maxsize=10000)
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
        else:
            return text
    except:
        return text  # fallback



import re
import nltk
import string
import wordninja
import inflect
from nltk.stem import PorterStemmer, WordNetLemmatizer
from word2number import w2n
from num2words import num2words
from collections import defaultdict

# Ensure required NLTK resources are available
try:
    nltk.download('wordnet', quiet=False)
    nltk.download('omw-1.4', quiet=False)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Please ensure you have a stable internet connection or download them manually.")
    print("You can try running: python -m nltk.downloader wordnet omw-1.4")

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
p = inflect.engine()

# Nickname dictionary (can be extended)
nickname_dict = {
    "bob": "robert", "bill": "william", "liz": "elizabeth",
    "mike": "michael", "alex": "alexander", "jim": "james",
    "billy": "william", "johnny": "john", "kate": "katherine"
}

# Character shape corrections
shape_corrections = {
    "0": "o", "1": "l", "3": "e", "5": "s", "rn": "m", "vv": "w",
    "13": "b", "7": "t"
}

def normalize_numbers(text):
    try:
        # Try convert English words to numbers (e.g., "one" -> "1")
        words = text.split()
        normalized_words = []
        for word in words:
            try:
                normalized_words.append(str(w2n.word_to_num(word)))
            except:
                # Try ordinal conversion (e.g., "1st" -> "first")
                if p.singular_noun(word):
                    normalized_words.append(p.singular_noun(word))
                else:
                    normalized_words.append(word)
        return " ".join(normalized_words)
    except:
        return text

def apply_shape_corrections(text):
    for wrong, right in shape_corrections.items():
        text = text.replace(wrong, right)
    return text

def apply_nickname_mapping(text):
    words = text.split()
    return " ".join([nickname_dict.get(w, w) for w in words])

def preprocess_variant(text):
    text = translate_to_english(text)
    text = text.lower()
    text = re.sub(r'[.,"\-/\'&()]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace(" ", "")
    text = apply_shape_corrections(text)
    text = normalize_numbers(text)
    words = wordninja.split(text)
    words = [ps.stem(w) for w in words]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [nickname_dict.get(w, w) for w in words]
    return " ".join(sorted(words))


import pandas as pd
import re
from tqdm import tqdm
import wordninja
import os
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch

# 文件列表
file_pairs = [
    (r'D:\code\data mining\alternate.csv', r'D:\code\data mining\alternate_preprocessed.csv'),
    # (r'D:\code\data mining\test_02\Sheet06.csv', r'D:\code\data mining\test_02\Sheet06_preprocessed.csv'),
    (r'D:\code\data mining\test_01.csv', r'D:\code\data mining\test_01_preprocessed.csv'),
    (r'D:\code\data mining\primary.csv', r'D:\code\data mining\primary_preprocessed.csv'),
]

# 预处理函数


# 执行预处理
def preprocess_files():
    for input_path, output_path in file_pairs:
        print(f"\n处理文件: {os.path.basename(input_path)}")
        try:
            df = pd.read_csv(input_path, encoding='utf-8-sig')
        except FileNotFoundError:
            print(f"❌ 错误：文件未找到：{input_path}")
            continue

        if 'NAME' in df.columns:
            print("开始预处理...")
            df['NAME'] = [preprocess_variant(name) for name in tqdm(df['NAME'], desc="预处理进度")]
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 预处理完成，已保存到：{output_path}")
        else:
            print(f"⚠️ 警告：文件中未找到 'NAME' 列：{input_path}")

preprocess_files()

# 读取预处理数据
primary_names = pd.read_csv('primary_preprocessed.csv')
alternate_names = pd.read_csv('alternate_preprocessed.csv')
# test_names = pd.read_csv('test_02/Sheet06_preprocessed.csv')
test_names = pd.read_csv('test_01_preprocessed.csv')

# 合并并去重
def merge_and_deduplicate(df1, df2):
    combined = pd.concat([df1[['NAME', 'ID']], df2[['NAME', 'ID']]], ignore_index=True)
    deduped = combined.drop_duplicates(subset=['NAME'], keep='first')
    return deduped

deduped_names = merge_and_deduplicate(primary_names, alternate_names)
primary_names_list = deduped_names['NAME'].tolist()
primary_ids = deduped_names['ID'].tolist()

# 加载模型
print("\n加载模型...")
embedder = SentenceTransformer(r'D:\code\data mining\all-MiniLM-L6-v2')
primary_embeddings = embedder.encode(primary_names_list, convert_to_tensor=True)

# 设置权重（你可以修改 x 和 y 的值）
x = 0.6  # semantic_score 权重 0.6
y = 0.4  # fuzzy_score 权重 0.4 68.4

# 匹配函数
def match_test_variant_combined(test_row):
    test_id = test_row['ID']
    test_variant = test_row['NAME']
    test_embedding = embedder.encode(test_variant, convert_to_tensor=True)

    # 计算语义相似度
    cosine_scores = util.pytorch_cos_sim(test_embedding, primary_embeddings)[0]

    best_score = -1
    best_id = None

    for idx, primary_name in enumerate(primary_names_list):
        semantic_score = cosine_scores[idx].item()
        fuzzy_score = fuzz.token_sort_ratio(test_variant, primary_name) / 100
        final_score = x * semantic_score + y * fuzzy_score

        if final_score > best_score:
            best_score = final_score
            best_id = primary_ids[idx]

    return test_id, best_id, best_score

# 执行评估
def evaluate_combined_matching(test_names):
    correct_matches = 0
    total_tests = len(test_names)

    for _, row in tqdm(test_names.iterrows(), total=total_tests, desc="匹配进度"):
        test_id, matched_id, score = match_test_variant_combined(row)
        if test_id == matched_id:
            correct_matches += 1

    accuracy = correct_matches / total_tests
    return accuracy

# 执行匹配
accuracy = evaluate_combined_matching(test_names)
print(f"\n综合评分匹配准确率: {accuracy * 100:.2f}%")
import pandas as pd
import re
import wordninja
from rapidfuzz import process, fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 数据预处理函数
def recover_and_segment(name):
    if pd.isna(name):
        return ''
    clean = re.sub(r'[^a-zA-Z0-9]', '', name)
    segments = wordninja.split(clean.lower())
    return ' '.join(sorted(segments))

def normalize_and_sort_words(name):
    clean = re.sub(r'[^a-zA-Z0-9]', '', name)
    segments = wordninja.split(clean.lower())
    return ' '.join(segments)

# 语言检测函数
def is_english(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

# 翻译函数(需要Google Cloud Translation API)
def translate_text(text, target_language='en'):
    """
    使用Google翻译API翻译文本
    需要先安装google-cloud-translate库并设置API密钥
    pip install google-cloud-translate
    """
    from google.cloud import translate_v2 as translate
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

# 分级匹配函数
def hierarchical_match(test_row, primary_names_list, primary_ids):
    test_id = test_row['ID']
    test_text = test_row['NAME']
    
    # 第一阶段：语言检测
    if not is_english(test_text):
        translated_text = translate_text(test_text)
        match, score, _ = process.extractOne(
            translated_text,
            primary_names_list,
            scorer=fuzz.WRatio
        )
        if score >= 90:
            matched_index = primary_names_list.index(match)
            return test_id, primary_ids[matched_index], score
    
    # 第二阶段：使用字典预处理
    from string_dicts import VISUAL_SIMILAR, ABBREVIATIONS, NICKNAMES
    
    def apply_dicts(text):
        # 应用视觉相似字符替换
        for k, v in VISUAL_SIMILAR.items():
            text = text.replace(k, v)
        # 应用缩写扩展
        words = text.split()
        words = [ABBREVIATIONS.get(word.lower(), word) for word in words]
        # 应用昵称标准化
        words = [NICKNAMES.get(word.lower(), word) for word in words]
        return ' '.join(words)
    
    processed_text = apply_dicts(test_text)
    match, score, _ = process.extractOne(
        processed_text,
        primary_names_list,
        scorer=fuzz.token_sort_ratio
    )
    if score >= 90:
        matched_index = primary_names_list.index(match)
        return test_id, primary_ids[matched_index], score
    
    # 第三阶段：使用原始模糊匹配
    return match_test_variant(test_row, primary_names_list, primary_ids)

# 原始匹配函数
def match_test_variant(test_row, primary_names_list, primary_ids):
    test_id = test_row['ID']
    test_variant_split = recover_and_segment(test_row['NAME'])
    test_variant = test_row['NAME']

    match1, score1, _ = process.extractOne(
        test_variant,
        primary_names_list,
        processor=lambda x: re.sub(r'[^a-z0-9]', '', x.lower()),
        scorer=fuzz.WRatio
    )

    match2, score2, _ = process.extractOne(
        test_variant_split,
        primary_names_list,
        processor=lambda x: re.sub(r'[^a-z0-9 ]', '', x.lower()),
        scorer=fuzz.token_sort_ratio
    )

    if score1 > score2:
        match = match1
        score = score1
    else:
        match = match2
        score = score2

    matched_index = primary_names_list.index(match)
    matched_id = primary_ids[matched_index]
    return test_id, matched_id, score

# 并行匹配主函数
def fuzzy_match_test_to_primary(test_names, primary_names):
    primary_names_list = primary_names['NAME'].tolist()
    primary_ids = primary_names['ID'].tolist()
    
    correct_matches = 0
    total_tests = len(test_names)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(match_test_variant, test_row, primary_names_list, primary_ids)
            for test_row in test_names.to_dict('records')
        ]

        for future in tqdm(as_completed(futures), total=total_tests, desc="匹配进度"):
            test_id, matched_id, score = future.result()
            if test_id == matched_id and score > 80:
                correct_matches += 1

    accuracy = correct_matches / total_tests
    return accuracy

# 测试案例函数
def run_test_cases():
    test_cases = [
        ("TRADINGLIMITEDGENERALALCARDINAL", "ALCARDINAL GENERAL TRiADING LIMITED"),
        ("PADIERNAPENALuisOrlando", "Luisg PENA, PADIERNAM Orlando"),
        ("G O L D E N S T A R C O", "GOLDEN TAR CO"),
        ("S H A N G H A I S E A X U A N W U L T D F R E I G H T Y O U B I C O", 
         "XUANWU YOUBI SEA FREI`HT SHANGHAI CO LTD"),
        ("HITTA, Sidan", "HITTA, Sidan Ag"),
        ("PREM INVESTMENT GROUP SAL (OFF-SHORE)", "PREMIER INVESTMENT GROUP SAL (OFF-SHORE)"),
        ("SHELESTENKO, Hennadiy O.", "SHELESTENKO, Hennadiy Oleksandrovych")
    ]
    
    for case1, case2 in test_cases:
        score_raw = fuzz.WRatio(case1, case2)
        score_norm = fuzz.token_sort_ratio(
            normalize_and_sort_words(case1),
            normalize_and_sort_words(case2)
        )
        score_partial = fuzz.partial_ratio(
            normalize_and_sort_words(case1),
            normalize_and_sort_words(case2)
        )
        
        print(f"\n测试案例: {case1} <-> {case2}")
        print(f"WRatio 分数: {score_raw}")
        print(f"Token Sort Ratio 分数: {score_norm}")
        print(f"Partial Ratio 分数: {score_partial}")

# 主程序
if __name__ == "__main__":
    # 运行测试案例
    print("运行测试案例...")
    run_test_cases()
    
    # 加载数据
    alternate_names = pd.read_csv('alternate.csv', nrows=1000)
    primary_names = pd.read_csv('primary.csv')
    
    # 读取Excel文件
    excel_file = "test_02.xlsx"
    sheets = pd.ExcelFile(excel_file).sheet_names
    sheets = sheets[1:]
    # 处理每个sheet
    for sheet in sheets:
        if sheet == "Sheet8":
            break
        df = pd.read_excel(excel_file, sheet_name=sheet)
        
        
        # 计算匹配准确率
        accuracy = fuzzy_match_test_to_primary(df, primary_names)
        print(f"Sheet {sheet} 准确率: {accuracy * 100:.2f}%")
import string
import html
import pandas as pd
import numpy as np
import re
import json
import os
import logging

from typing import List
from hashlib import md5
from typing import Any, List


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except:
        return {}


def write_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def save_triplets_to_txt(triplets, file_path):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"{triplets[0]},{triplets[1]},{triplets[2]}\n")


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    calculate cosine similarity between two vectors
    """
    dot_product = np.dot(vector1, vector2)
    magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if not magnitude:
        return 0
    return dot_product / magnitude


def create_file_if_not_exists(file_path: str):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")


def update_df_with_new_ids(original_df, new_data):
    """
    高效更新DataFrame，仅添加新ID的数据行

    参数：
    original_df : 原始DataFrame，需包含'id'列
    new_data    : 新数据，支持字典列表/DataFrame

    返回：
    (updated_df, added_count)
    updated_df   : 更新后的DataFrame
    added_count  : 实际新增的行数
    """
    # 将新数据转换为DataFrame
    new_df = pd.DataFrame(new_data)

    # 高效过滤逻辑（使用集合加速查询）
    existing_ids = set(original_df["id"])
    new_ids = set(new_df["id"])

    # 计算需要添加的唯一ID集合
    unique_new_ids = new_ids - existing_ids

    # 二次过滤避免新数据中的重复ID
    mask = (new_df["id"].isin(unique_new_ids)) & (~new_df.duplicated("id"))
    filtered_df = new_df[mask]

    # 合并数据
    updated_df = pd.concat([original_df, filtered_df], ignore_index=True)

    return updated_df, len(filtered_df)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]

    return result


async def handle_single_entity_extraction(
    record_attributes: list[str],
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    return dict(
        entity_name=entity_name, entity_type=entity_type, description=entity_description
    )


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


async def handle_single_relationship_extraction(record_attributes: list[str]):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    edge_description = clean_str(record_attributes[2])
    target = clean_str(record_attributes[3].upper())
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source, tgt_id=target, weight=weight, description=edge_description
    )


def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if "." in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist


def extract_values_from_json(
    json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False
):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group("key").strip('"')  # Strip quotes from key
        value = match.group("value").strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith("{") and value.endswith("}"):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logging.warning("No values could be extracted from the string.")

    return extracted_values


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logging.error(
                            f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}..."
                        )
                        return None
                    finally:
                        first_json_start = None
    logging.warning("No complete JSON object found in the input string.")
    return None


def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    prediction_json = extract_first_complete_json(response)

    if prediction_json is None:
        logging.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)

    if not prediction_json:
        logging.error("Unable to extract meaningful data from the response.")
    else:
        logging.info("JSON data successfully extracted.")

    return prediction_json


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data]
    )


def generate_doc_id(file_path: str):
    name = os.path.basename(file_path)
    return compute_mdhash_id(name, prefix="doc-")


def policy_docs_parser(file_paths: List[str]):
    items = []
    for path in file_paths:
        name = os.path.basename(path)
        doc_id = generate_doc_id(path)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        items.append({"document_id": doc_id, "name": name, "content": content})
    new_df = pd.DataFrame(items)
    return new_df


def policy_docs_parser_by_content(file_name: str, content: str):
    items = []
    doc_id = generate_doc_id(file_name)
    items.append({"document_id": doc_id, "name": file_name, "content": content})
    new_df = pd.DataFrame(items)
    return new_df


async def split_text_by_file(
    file_path: str, segment_length: int, overlap_length: int
) -> List:
    """
    将文本文件分割成多个片段，每个片段的长度为segment_length，相邻片段之间有overlap_length的重叠。

    参数:
    - file_path: 文本文件的路径
    - segment_length: 每个片段的长度，默认为300
    - overlap_length: 相邻片段之间的重叠长度，默认为50

    返回:
    - 包含片段ID和片段内容
    """
    file_name = os.path.basename(file_path).split(".")[0]
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()  # 读取文件内容
    return split_text_by_str(file_name, content, segment_length, overlap_length)


async def split_text_by_str(
    file_name: str, content: str, segment_length: int, overlap_length: int
) -> List:
    """
    将文本文件分割成多个片段，每个片段的长度为segment_length，相邻片段之间有overlap_length的重叠。

    参数:
    - content: 文本内容
    - segment_length: 每个片段的长度，默认为300
    - overlap_length: 相邻片段之间的重叠长度，默认为50

    返回:
    - 包含片段ID和片段内容
    """
    chunks = []
    document_id = generate_doc_id(file_name)

    text_segments = []  # 用于存储分割后的文本片段
    start_index = 0  # 初始化起始索引

    # 循环分割文本，直到剩余文本长度不足以形成新的片段
    while start_index + segment_length <= len(content):
        text_segments.append(content[start_index : start_index + segment_length])
        start_index += segment_length - overlap_length  # 更新起始索引，考虑重叠长度

    # 处理剩余的文本，如果剩余文本长度小于segment_length但大于0
    if start_index < len(content):
        text_segments.append(content[start_index:])

    # 为每个片段生成唯一的ID，并将其存储在字典中
    for segement in text_segments:
        document_id
        chunks.append(
            {
                "id": compute_mdhash_id(segement, prefix="chunk-"),
                "file_name": file_name,
                "content": segement,
                "document_id": document_id,
            }
        )

    return chunks


async def split_text_by_sentence(
    file_name: str, content: str, segment_length: int = 300, overlap_length: int = 50
) -> List[dict]:
    """
    按 '。' 分割文本，并将过短的句子合并，最后得到指定长度的文本片段。

    参数:
    - file_name: 文件名
    - content: 文本内容
    - segment_length: 每个片段的最大长度
    - overlap_length: 相邻片段之间的重叠长度

    返回:
    - 包含片段ID和片段内容的列表
    """
    chunks = []
    document_id = generate_doc_id(file_name)

    # 先按 '。' 切分，再补回分隔符
    sentences = [s.strip() + "。" for s in content.split("。") if s.strip()]

    text_segments = []
    current_chunk = ""

    # 检查sentences中的句子，如果长度大于segment_length，就切割
    for i, sentence in enumerate(sentences):
        if len(sentence) <= segment_length:
            text_segments.append(sentence)
        else:
            start_index = 0  # 初始化起始索引

            # 循环分割文本，直到剩余文本长度不足以形成新的片段
            while start_index + segment_length <= len(sentence):
                text_segments.append(
                    sentence[start_index : start_index + segment_length]
                )
                start_index += segment_length

    sentences = text_segments
    text_segments = []

    for sentence in sentences:
        # 如果当前chunk加上句子后还不超长，就继续加
        if len(current_chunk) + len(sentence) <= segment_length:
            current_chunk += sentence
        else:
            # 保存当前chunk
            text_segments.append(current_chunk)
            # 开启新的chunk
            current_chunk = sentence

    # 把最后一个未保存的chunk加进去
    if current_chunk:
        text_segments.append(current_chunk)

    # 处理 overlap
    final_chunks = []
    for i, seg in enumerate(text_segments):
        start = max(0, i - 1)
        # 给每个chunk前面拼接一部分 overlap
        if i > 0 and overlap_length > 0:
            overlap_part = text_segments[start][-overlap_length:]
            seg = overlap_part + seg
        final_chunks.append(seg)

    # 生成唯一 ID
    for seg in final_chunks:
        chunks.append(
            {
                "id": compute_mdhash_id(seg, prefix="chunk-"),
                "file_name": file_name,
                "content": seg,
                "document_id": document_id,
            }
        )

    return chunks


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len((key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


def normalize_text(text: str) -> str:
    """
    将文本中的标点符号统一为英文标点，并过滤掉不支持的符号。

    该函数会执行以下操作：
    1. 将常见的中文标点符号替换为对应的英文标点符号。
    2. 将所有非（汉字、英文字母、阿拉伯数字、英文标点和空格）的字符替换为空格。
    3. 移除多余的空格，确保单词和字符之间只有一个空格。

    Args:
        text: 需要处理的原始字符串。

    Returns:
        处理完成的字符串。
    """
    if not isinstance(text, str):
        raise TypeError("输入必须是字符串")

    # 1. 定义中英文标点符号映射表
    punctuation_mapping = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "—": "-",
        "～": "~",
        "、": ",",
        "…": "...",
        "〔": "[",
        "〕": "]",
    }

    # 2. 替换中文标点为英文标点
    for zh_punc, en_punc in punctuation_mapping.items():
        text = text.replace(zh_punc, en_punc)

    # 3. 将所有非（汉字、英文字母、阿拉伯数字、基本英文标点和空格）的字符替换为空格
    # 保留的字符范围：
    # \u4e00-\u9fa5: 汉字
    # a-zA-Z: 英文字母
    # 0-9: 阿拉伯数字
    # string.punctuation: 英文标点符号 (e.g., !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    # \s: 空格、制表符等空白符
    allowed_chars = r"[^\u4e00-\u9fa5a-zA-Z0-9" + re.escape(string.punctuation) + r"\s]"
    text = re.sub(allowed_chars, " ", text)

    # 4. 移除多余的空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_values_recursive(json_content_string, keys=None):
    """
    [内部辅助函数] 使用正则表达式递归地从 JSON 内容字符串中提取键值。
    这个函数假定传入的字符串是 JSON 对象的内容（即不包含最外层的花括号）。
    """
    extracted_values = {}
    # 增强版正则表达式，能更好地处理值中的转义引号 \"
    regex_pattern = r'(?P<key>"?[\w\.]+"?)\s*:\s*(?P<value>{.*?}|\[.*?\]|".*?(?<!\\)"|\'.*?(?<!\\)\'|[^,}\]]+)'

    # 使用 re.finditer 来处理多个顶层键值对
    matches = list(re.finditer(regex_pattern, json_content_string, re.DOTALL))

    for match in matches:
        key = match.group("key").strip("'\"")

        if keys is not None and key not in keys:
            continue

        value_str = match.group("value").strip()

        # --- 这是关键的修复点 ---
        # 对于字典和列表，我们传入其 *内部* 的内容进行递归

        if value_str.startswith("{") and value_str.endswith("}"):
            # 递归解析字典内容
            extracted_values[key] = _extract_values_recursive(
                value_str[1:-1], keys=None
            )  # 在子结构中提取所有键
        elif value_str.startswith("[") and value_str.endswith("]"):
            list_items = []
            list_content = value_str[1:-1].strip()
            # 查找列表中的所有字典对象。这个正则对于简单列表有效
            item_matches = re.finditer(r"({.*?})", list_content, re.DOTALL)
            for item_match in item_matches:
                dict_str = item_match.group(1)
                # 递归解析每个字典项的内容
                parsed_dict = _extract_values_recursive(dict_str[1:-1], keys=None)
                if parsed_dict:
                    list_items.append(parsed_dict)
            extracted_values[key] = list_items
        else:
            extracted_values[key] = parse_value(value_str)

    return extracted_values


def robust_json_parser(text_blob, keys=None):
    """
    一个健壮的 JSON 解析器。
    它首先尝试使用标准库 json 解析。如果失败，则回退到自定义的正则表达式解析器。

    Args:
        text_blob (str): 包含类 JSON 数据的字符串，可能包含代码块标记。
        keys (list, optional): 希望在顶层提取的键的列表。如果为 None，则提取所有。

    Returns:
        dict: 解析后的字典。
    """
    # 1. 清理输入字符串，移除 markdown 代码块标记
    clean_str = text_blob.strip()
    if clean_str.startswith("```json"):
        clean_str = clean_str[7:]
    elif clean_str.startswith("```"):
        clean_str = clean_str[3:]
    if clean_str.endswith("```"):
        clean_str = clean_str[:-3]
    clean_str = clean_str.strip()

    # 2. 优先尝试使用标准库解析
    try:
        parsed_json = json.loads(clean_str)
        # 如果指定了 keys，则从解析结果中筛选
        if keys is not None:
            return {k: v for k, v in parsed_json.items() if k in keys}
        return parsed_json
    except json.JSONDecodeError as e:
        logging.warning(f"标准 JSON 解析失败: {e}. 回退到正则表达式解析器。")
        # 3. 如果标准库失败，使用我们自定义的备用方案
        # 确保我们只传入 JSON 对象的内容给递归函数
        if clean_str.startswith("{") and clean_str.endswith("}"):
            content = clean_str[1:-1]
            return _extract_values_recursive(content, keys=keys)
        else:
            # 如果字符串不是一个完整的对象，也尝试直接解析
            return _extract_values_recursive(clean_str, keys=keys)

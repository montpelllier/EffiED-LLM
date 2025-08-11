"""
表格错误模拟器模块
用于模拟各种类型的数据错误，包括typo、rule violation、pattern violation等
"""

import json
import random
import string
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd


def create_typo_replacements():
    """
    创建一个包含常见typo错误的替换字典
    包括键盘布局错误、视觉相似字符、常见拼写错误等
    """

    replacements = {}

    # === 键盘布局相邻按键错误（QWERTY布局）===
    keyboard_neighbors = {
        # 第一行
        'q': ['w', 'a', 's'], 'w': ['q', 'e', 'a', 's', 'd'],
        'e': ['w', 'r', 's', 'd', 'f'], 'r': ['e', 't', 'd', 'f', 'g'],
        't': ['r', 'y', 'f', 'g', 'h'], 'y': ['t', 'u', 'g', 'h', 'j'],
        'u': ['y', 'i', 'h', 'j', 'k'], 'i': ['u', 'o', 'j', 'k', 'l'],
        'o': ['i', 'p', 'k', 'l'], 'p': ['o', 'l'],

        # 第二行
        'a': ['q', 'w', 's', 'z', 'x'], 's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
        'd': ['w', 'e', 'r', 's', 'f', 'x', 'c', 'v'], 'f': ['e', 'r', 't', 'd', 'g', 'c', 'v', 'b'],
        'g': ['r', 't', 'y', 'f', 'h', 'v', 'b', 'n'], 'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n', 'm'],
        'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm'], 'k': ['u', 'i', 'o', 'j', 'l', 'm'],
        'l': ['i', 'o', 'p', 'k'],

        # 第三行
        'z': ['a', 's', 'x'], 'x': ['a', 's', 'd', 'z', 'c'],
        'c': ['s', 'd', 'f', 'x', 'v'], 'v': ['d', 'f', 'g', 'c', 'b'],
        'b': ['f', 'g', 'h', 'v', 'n'], 'n': ['g', 'h', 'j', 'b', 'm'],
        'm': ['h', 'j', 'k', 'n']
    }

    # 添加键盘相邻错误
    for char, neighbors in keyboard_neighbors.items():
        replacements[char] = neighbors + [char.upper()]  # 包含大小写切换
        replacements[char.upper()] = [n.upper() for n in neighbors] + [char]

    # === 视觉相似字符错误 ===
    visual_similar = {
        # 数字与字母
        '0': ['O', 'o', 'Q'], '1': ['l', 'I', '|'], '2': ['Z'], '5': ['S'], '6': ['G'], '8': ['B'],
        'O': ['0', 'Q', 'o'], 'o': ['0', 'O'], 'I': ['1', 'l', '|'], 'l': ['1', 'I', '|'],
        'S': ['5', '$'], 'G': ['6'], 'B': ['8'], 'Z': ['2'],

        # 相似形状的字母
        'm': ['rn', 'nn'], 'w': ['vv'],
        'P': ['R'], 'R': ['P'], 'H': ['N'], 'N': ['H'],
        'u': ['n'], 'n': ['u', 'h'], 'h': ['n', 'b'], 'b': ['h', 'd', 'p'],
        'd': ['b', 'p', 'q', 'cl'], 'p': ['b', 'd', 'q'], 'q': ['p', 'd', 'g'],

        # 特殊字符
        '|': ['1', 'I', 'l'], '/': ['\\'], '\\': ['/'],
        '.': [','], ',': ['.'], ';': [':'], ':': [';'],
        '"': ["'"], "'": ['"'], '`': ["'"], '~': ['-', '_'],
    }

    # 添加视觉相似错误
    for char, similars in visual_similar.items():
        if char in replacements:
            replacements[char].extend(similars)
        else:
            replacements[char] = similars[:]

    # === 符号键盘错误（Shift + 数字）===
    symbol_errors = {
        '!': ['1', '@'], '@': ['2', '!', '#'], '#': ['3', '@', '$'],
        '$': ['4', '#', '%'], '%': ['5', '$', '^'], '^': ['6', '%', '&'],
        '&': ['7', '^', '*'], '*': ['8', '&', '('], '(': ['9', '*', ')'],
        ')': ['0', '(', '_'], '_': ['-', ')'], '-': ['_', '='], '=': ['-', '+'],
        '+': ['='], '[': ['{', ']'], ']': ['[', '}'], '{': ['['], '}': [']'],
        ';': [':', "'"], ':': [';'], "'": ['"', ';'], '"': ["'"]
    }

    for symbol, errors in symbol_errors.items():
        if symbol in replacements:
            replacements[symbol].extend(errors)
        else:
            replacements[symbol] = errors[:]

    # === 清理和去重 ===
    for char in replacements:
        # 去除重复项
        replacements[char] = list(set(replacements[char]))
        # 去除自己
        if char in replacements[char]:
            replacements[char].remove(char)
    return replacements


def replace_characters(text: str, replacements: Dict[str, Union[str, List[str]]]) -> str:
    """
    随机替换字符串中的字符

    Args:
        text: 待替换的字符串
        replacements: 字符替换映射字典，格式为 {原字符: 新字符或新字符列表}
                     如果字符不在字典中，将从ASCII可打印字符中随机选择

    Returns:
        替换后的字符串
    """
    if not text or not replacements:
        return text

    # 所有位置都可以替换
    if len(text) == 0:
        return text

    # 随机选择第i位进行替换
    random_position = random.randint(0, len(text) - 1)
    char_to_replace = text[random_position]

    # 如果字符在replacements中，使用指定的替换；否则从ASCII中随机选择
    if char_to_replace in replacements:
        replacement_options = replacements[char_to_replace]
        # 如果是列表，随机选择一个；如果是字符串，直接使用
        if isinstance(replacement_options, list):
            replacement_char = random.choice(replacement_options)
        else:
            replacement_char = replacement_options
    else:
        # 从ASCII可打印字符中随机选择（字母、数字、标点符号等）
        # ASCII 33-126 包含所有可打印字符（除空格外）
        # ASCII 32 是空格，我们也包含进来
        ascii_chars = [chr(i) for i in range(32, 127)]
        replacement_char = random.choice(ascii_chars)

    # 替换第i位的字符
    text_list = list(text)
    text_list[random_position] = replacement_char
    return ''.join(text_list)

def introduce_typo(text: str, strategy: str) -> str:
    """
    在字符串中随机引入拼写错误

    Args:
        text: 待处理的字符串
        strategy: 错误引入策略，可以是 'replace'（替换字符）、'insert'（插入字符）、'delete'（删除字符）、

    Returns:
        包含拼写错误的字符串
    """
    if not text:
        return text
    if strategy == 'all':
        strategy = random.choice(['replace', 'insert', 'delete', 'transpose', 'empty'])

    if strategy == 'replace':
        # 替换字符
        replacements = create_typo_replacements()
        return replace_characters(text, replacements)
    elif strategy == 'insert':
        # 插入字符
        pos = random.randint(0, len(text))
        char_pos = random.choice([max(0, pos - 1), min(len(text) - 1, pos)])
        # char = text
        return text[:pos] + text[char_pos] + text[pos:]
    elif strategy == 'delete':
        # 删除字符
        if len(text) > 1:
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]
    elif strategy == 'transpose':
        # 交换字符
        if len(text) > 1:
            pos = random.randint(0, len(text) - 2)
            return text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]
    elif strategy == 'empty':
        # 设置为空值
        return set_empty_value(text)

    return text


def set_empty_value(value: str) -> pd.DataFrame:
    return random.choice(['empty', 'null', 'N/A', 'NaN', '']) if value else value


def violate_functional_dependencies(dataframe: pd.DataFrame, columns: list[str], error_rate: float) -> pd.DataFrame:
    """
    违反函数依赖规则，破坏数据的一致性约束

    Args:
        dataframe: 原始数据框
        dependency_rules: 函数依赖规则字典，格式为 {决定属性: [被决定属性列表]}
                         例如: {'employee_id': ['name', 'department'], 'department': ['manager']}
        error_rate: 错误率（0-1之间）

    Returns:
        包含错误的数据框和错误位置列表
    """
    df_copy = dataframe.copy()

    for col in columns:
        if col not in df_copy.columns:
            continue

        unique_vals = df_copy[col].dropna().unique()
        if len(unique_vals) <= 1:
            # 如果只有一个唯一值或没有值，跳过这列
            continue

            # 计算要修改的行数
        total_rows = len(df_copy)
        num_errors = int(total_rows * error_rate)

        if num_errors == 0:
            continue

        available_indices = df_copy.index.tolist()
        indices_to_modify = random.sample(available_indices, min(num_errors, len(available_indices)))
        for idx in indices_to_modify:
            if pd.notna(df_copy.loc[idx, col]):
                # 获取当前值
                current_value = df_copy.loc[idx, col]

                # 从其他值中选择一个不同的值进行替换
                other_values = [val for val in unique_vals if val != current_value]

                if len(other_values) > 0:
                    # 随机选择一个不同的值
                    replacement_value = random.choice(other_values)
                    df_copy.loc[idx, col] = replacement_value
    return df_copy




def main():
    clean_data = pd.read_csv('D:\Programming\Python\MasterProject\data\spotify_history.csv', dtype=str, na_values=[], keep_default_na=False)
    # print(clean_data.columns)
    print(len(clean_data))

    clean_data = clean_data[:10000]
    typo_columns = ['ts', 'platform', 'ms_played', 'track_name',
       'artist_name', 'album_name', 'reason_start', 'reason_end', 'shuffle',
       'skipped']
    fd_columns = ['track_name', 'artist_name', 'album_name']
    fd_df = violate_functional_dependencies(clean_data, fd_columns, 0.01)

    for col in typo_columns:
        sample_indices = random.sample(range(len(fd_df)), random.randint(100, 2000))
        for idx in sample_indices:
            fd_df.at[idx, col] = introduce_typo(str(fd_df.at[idx, col]), 'all')
    clean_data.to_csv('D:\Programming\Python\MasterProject\data\spotify_history_clean.csv', index=False)
    fd_df.to_csv('D:\Programming\Python\MasterProject\data\spotify_history_error.csv', index=False)
    is_err_df = clean_data != fd_df
    for col in is_err_df:
        print(f"{col} has {is_err_df[col].sum()} errors, {is_err_df[col].sum() / len(is_err_df) * 100:.2f}% of total rows")

if __name__ == "__main__":
    main()

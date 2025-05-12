import re

import pandas as pd


def extract_function_name(func_str):
    """
    提取 def 后的函数名
    """
    match = re.search(r'^\s*def\s+(\w+)\s*\(', func_str)
    return match.group(1) if match else None

def parse_functions(func_str_list):
    """
    将函数定义字符串转换为函数对象列表。
    """
    funcs = {}
    for i, func_str in enumerate(func_str_list):
        local_vars = {}
        try:
            exec(func_str, {}, local_vars)
            func = next((v for v in local_vars.values() if callable(v)), None)
            name = extract_function_name(func_str) or f'func_{i}'

            if not func:
                raise ValueError(f"No callable found in function {i}")

            if name in funcs:
                raise ValueError(f"Duplicate function name: {name}")

            funcs[name] = func

        except Exception as e:
            print(f"Error parsing function {i}: {func_str}")
            raise ValueError(f"Error: {e}")
    return funcs


class FunctionVectorizer:
    def __init__(self, function_list: list[str]):
        """
        初始化函数向量器，传入函数定义的字符串列表。
        每个字符串应定义一个函数 def f(x): ...
        """
        self.func_list = function_list
        self.funcs = parse_functions(function_list)

    def transform(self, series):
        """
        将所有函数应用于 Pandas Series，返回布尔矩阵 DataFrame。
        遇到错误时，返回错误字符串。自动移除恒定列。
        """
        def safe_apply(func, x):
            try:
                return func(x)
            except Exception as e:
                print(f"Error applying function {func.__name__} to {x}: {e}")
                return str(e)  # or return False / None depending on preference

        # 应用每个函数
        df_result = pd.DataFrame({
            name: series.apply(lambda x: safe_apply(func, x))
            for name, func in self.funcs.items()
        }, index=series.index)
        print(df_result)

        # 过滤掉恒定列（所有值相同）
        return df_result.loc[:, ~df_result.nunique(dropna=False).eq(1)]


if __name__ == "__main__":
    func_list = [
        "def f1(x): return x > 0",
        "def f2(x): return x % 2 == 0",
        "def f3(x): return x < 100",
        "def f4(x): return x.list()"
    ]

    df = pd.DataFrame({'col': [1, -2, 100, 50]})

    fv = FunctionVectorizer(func_list)
    vector = fv.transform(df['col'])

    print(vector)

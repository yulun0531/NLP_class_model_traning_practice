import re

# 正則表達式
pattern = r'[a-z].*b$'

# 測試的字串
strings = ["cat1b", "dog", "lowercaseb2", "1ab", "23b1", "ebd"]

# 使用正則表達式進行測試
for string in strings:
    if re.match(pattern, string):
        print(f"{string}: 符合條件")
    else:
        print(f"{string}: 不符合條件")

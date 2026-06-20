"""
Formula/LaTeX post-processing utilities.

Normalizes LaTeX strings by removing unnecessary spaces, handling special
commands (\operatorname, \mathrm, \text, \mathbf), and cleaning up
artifacts from the tokenizer.
"""

import re


def post_process_formula(s: str) -> str:
    """Post-process LaTeX string: normalize spacing and handle special commands.

    Args:
        s: Raw LaTeX string from tokenizer.

    Returns:
        Cleaned-up LaTeX string.
    """
    # Remove special tokens
    s = s.replace('[BOS]', '').replace('[EOS]', '').replace('[PAD]', '').strip()
    s = s.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()

    # Handle special LaTeX commands
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = "[\W_^\d]"
    names = []

    for x in re.findall(text_reg, s):
        pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
        matches = re.findall(pattern, x[0])
        for m in matches:
            if (
                m
                not in [
                    "\\operatorname",
                    "\\mathrm",
                    "\\text",
                    "\\mathbf",
                ]
                and m.strip() != ""
            ):
                s = s.replace(m, m + "XXXXXXX")
                s = s.replace(" ", "")
                names.append(s)

    if len(names) > 0:
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)

    news = s
    while True:
        s = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == s:
            break

    return s.replace("XXXXXXX", " ").strip()


def remove_chinese_text_wrapping(formula: str) -> str:
    """Remove \\text{} wrapping around Chinese characters.

    Args:
        formula: LaTeX formula string.

    Returns:
        Formula with Chinese text unwrapped.
    """
    pattern = re.compile(r"\\text\s*{\s*([^}]*?[一-鿿]+[^}]*?)\s*}")
    return pattern.sub(r"\1", formula).replace('"', "")

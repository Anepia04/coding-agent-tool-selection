"""文本工具（存在代码风格问题）。"""


def normalize( text ):
    """合并连续空白。"""
    return   " ".join( text.split() )

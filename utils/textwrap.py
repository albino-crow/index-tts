import re
import math

from openai import OpenAI

from settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def resize_sentence(text: str, percent: float, mode: str, language: str):
    """
    text: sentence or partial sentence
    percent: desired expansion/shrink percentage
    mode: "longer" or "shorter"
    language: the language of the input text (e.g., "Chinese", "English", "Farsi")
    """
    percent *= 100  # convert to percentage for prompt
    if language == "english":
        system_prompt = (
            "You are a precise text-resizer. "
            "You always keep the meaning of a sentence exactly the same. "
            "You must rewrite text only in the specified language. "
            "Do NOT add or remove meaning. "
            "Only adjust the length by approximately the requested percentage."
            "Do NOT use abbreviations; write out at least one full word instead of any shortened form."
        )
        user_prompt = f"""
            The following text is in **{language}**:

            {text}

            Task:
            Make it {mode.value} by about {percent:.2f} percent.
            Keep the SAME meaning.
            Keep it in **{language}**.
            NOTE: EXPECT FROM NAME ALL THE WORDS MUST BE IN {language}. NOT SINGLE WORD SHOULD BE IN OTHER LANGUAGE.
            Return ONLY the rewritten text.
            
            """

    else:
        system_prompt = (
            "你是一位精确的文本长度调节器。"
            "你始终保持句子的含义完全相同。"
            "你必须只用指定的语言改写文本。"
            "不要添加或删除任何含义。"
            "只根据请求的大约百分比调整文本长度。"
            "不要使用缩写；任何缩略形式都必须写成至少一个完整的词语。"
        )

        user_prompt = f"""
            以下文本为 **{language}**：

            {text}

            任务：
            将其缩放 {mode.value} 约 {percent:.2f}%。
            保持相同含义。
            保持使用 **{language}**。
            注意：期望输出中，从名称开始的所有词语必须使用 {language}。不能有任何一个词使用其他语言。

            只返回改写后的文本。
            """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def is_chinese(ch):
    return bool(re.match(r"[\u4e00-\u9fff]", ch))


def split_sentence_by_custom_ratios_preserved_ch(text, ratios):
    # Count total Chinese characters only
    chinese_chars = [ch for ch in text if is_chinese(ch)]
    total_syllables = len(chinese_chars)

    token_bucket = [[] for _ in range(len(ratios))]
    bucket_indice = 0
    count_sylabe = 0
    ratio = ratios[0]

    for token in text:
        # only count Chinese chars toward the limit
        if is_chinese(token):
            if (
                len(token_bucket[bucket_indice]) > 0
                and (count_sylabe / total_syllables) * 100 >= ratio
                and bucket_indice < len(ratios) - 1
            ):
                bucket_indice += 1
                ratio += ratios[bucket_indice]

            count_sylabe += 1
            token_bucket[bucket_indice].append(token)
        else:
            token_bucket[bucket_indice].append(token)

    return ["".join(tokens) for tokens in token_bucket]


def estimate_syllables_per_word(word):
    return (math.ceil(len(word) / 3) + 1.67) / 2


def estimate_total_syllables_combined(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    word_count = len(words)
    if word_count == 0:
        return 0

    # Average the two estimates
    final_syllable_estimate = sum(estimate_syllables_per_word(word) for word in words)

    return final_syllable_estimate


def split_sentence_by_custom_ratios_preserved_en(sentence, ratios):
    all_tokens = [token for token in re.split(r"(\b\w+\b)", sentence) if token]

    total_syllables = estimate_total_syllables_combined(sentence)

    token_bucket = [[] for _ in range(len(ratios))]
    bucket_indice = 0
    count_sylabe = 0
    ratio = ratios[0]
    for token in all_tokens:
        if re.match(r"\b\w+\b", token):
            if (
                len(token_bucket[bucket_indice]) > 0
                and (count_sylabe / total_syllables) * 100 >= ratio
                and bucket_indice < len(ratios) - 1
            ):
                bucket_indice += 1
                ratio += ratios[bucket_indice]

            count_sylabe += estimate_syllables_per_word(token)
            token_bucket[bucket_indice].append(token)

        else:
            token_bucket[bucket_indice].append(token)

    return ["".join(tokens) for tokens in token_bucket]

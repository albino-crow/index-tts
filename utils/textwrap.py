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

    system_prompt = (
        "You are a precise text-resizer. "
        "You always keep the meaning of a sentence exactly the same. "
        "You must rewrite text only in the specified language. "
        "Do NOT add or remove meaning. "
        "Only adjust the length by approximately the requested percentage."
    )

    user_prompt = f"""
The following text is in **{language}**:

{text}

Task:
Make it {mode} by about {percent}%.
Keep the SAME meaning.
Keep it in **{language}**.
Return ONLY the rewritten text.
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
    print(words)
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

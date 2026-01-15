"""
Generate circuits for the math case study. Excludes edges.
"""

import os
from pathlib import Path
from circuits.analysis.circuit_ops import Circuit
from util.subject import Subject, llama31_8B_instruct_config

from env_util import ENV
if ENV.ARTIFACTS_DIR is None:
    raise RuntimeError("ARTIFACTS_DIR must be set in the .env file")

ARTIFACTS_DIR = Path(ENV.ARTIFACTS_DIR) / "multilingual_circuit_hypotheses"

subject = Subject(llama31_8B_instruct_config)
tokenizer = subject.tokenizer

lang_to_prompt = {
    "en": "What is the opposite of {x}?",
    "zh": "{x}的反义词是什么？",
    "fr": "Quel est le contraire de {x} ?",
    "de": "Was ist das Gegenteil von {x}?",
    "es": "¿Cuál es el opuesto de {x}?",
    "it": "Qual è il contrario di {x}?",
    "ru": "Что является антонимом слова {x}?",
    "hi": "{x} का विपरीतार्थक शब्द क्या है?",
    "ar": "ما هو عكس كلمة {x}؟",
}
lang_to_seed_response = {
    "en": "Answer:",
    "zh": "答案：",
    "fr": "Réponse :",
    "de": "Antwort:",
    "es": "Respuesta:",
    "it": "Risposta:",
    "ru": "Ответ:",
    "hi": "उत्तर:",
    "ar": "الإجابة:",
}
word_to_lang = {
    "big": {
        "en": "big",
        "zh": "大",
        "fr": "grand",
        "de": "groß",
        "es": "grande",
        "it": "grande",
        "ru": "большой",
        "hi": "बड़ा",
        "ar": "كبير",
    },
    "small": {
        "en": "small",
        "zh": "小",
        "fr": "petit",
        "de": "klein",
        "es": "pequeño",
        "it": "piccolo",
        "ru": "маленький",
        "hi": "छोटा",
        "ar": "صغير",
    },
    "fast": {
        "en": "fast",
        "zh": "快",
        "fr": "rapide",
        "de": "schnell",
        "es": "rápido",
        "it": "veloce",
        "ru": "быстрый",
        "hi": "तेज़",
        "ar": "سريع",
    },
    "slow": {
        "en": "slow",
        "zh": "慢",
        "fr": "lent",
        "de": "langsam",
        "es": "lento",
        "it": "lento",
        "ru": "медленный",
        "hi": "धीमा",
        "ar": "بطيء",
    },
    "hot": {
        "en": "hot",
        "zh": "热",
        "fr": "chaud",
        "de": "heiß",
        "es": "caliente",
        "it": "caldo",
        "ru": "горячий",
        "hi": "गर्म",
        "ar": "حار",
    },
    "cold": {
        "en": "cold",
        "zh": "冷",
        "fr": "froid",
        "de": "kalt",
        "es": "frío",
        "it": "freddo",
        "ru": "холодный",
        "hi": "ठंडा",
        "ar": "بارد",
    },
}
word_to_axis = {
    "big": "size",
    "small": "size",
    "fast": "speed",
    "slow": "speed",
    "hot": "temperature",
    "cold": "temperature",
}
assert set(word_to_axis.keys()) == set(word_to_lang.keys())

prompts, seed_responses, labels = [], [], []
for lang, prompt_template in lang_to_prompt.items():
    for word, lang_to_word in word_to_lang.items():
        if lang in lang_to_word:
            prompts.append(prompt_template.format(x=lang_to_word[lang]))
            seed_responses.append(lang_to_seed_response[lang])
            labels.append(f"{word}___{lang}")


def main():
    print(f"Generated {len(prompts)} prompts")

    # convert to dataframes
    circuit = Circuit.from_dataset(
        subject,
        prompts,
        seed_responses,
        labels,
        return_nodes_only=False,
        neurons=None,
        percentage_threshold=0.005,
        batch_size=1,
        verbose=False,
        k=5,
        apply_blacklist=True,
    )

    os.makedirs(ARTIFACTS_DIR / "case_studies", exist_ok=True)
    circuit.save_to_pickle(ARTIFACTS_DIR / "case_studies" / "multilingual_circuit.pkl")


if __name__ == "__main__":
    main()

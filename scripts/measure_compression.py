"""Measure the bytes-per-patch compression ratio B_subword implies, per domain.

The compression ratio is the knob Bolmo uses for its efficiency experiments, and it
is a property of the *source tokenizer*, not of byteification. Measuring it tells
us what a byteified model inherits for free, and where the gaps are: the paper's
argument is that byteification matters most where subword tokenization is weak
(code, non-Latin scripts, biological sequences).
"""

from __future__ import annotations

import statistics

from transformers import AutoTokenizer

from byteify.alignment import align, build_token_byte_table

SAMPLES = {
    "english prose": "The quick brown fox jumps over the lazy dog. " * 20,
    "english news": (
        "The central bank raised interest rates again on Tuesday, citing persistent "
        "inflation across energy and housing markets, as policymakers signalled that "
        "borrowing costs would remain elevated through the end of the year. " * 6
    ),
    "code (python)": (
        "def compute_loss(predictions, targets, reduction='mean'):\n"
        "    mask = targets != -100\n"
        "    loss = functional.cross_entropy(predictions, targets, reduction='none')\n"
        "    return loss[mask].mean() if reduction == 'mean' else loss\n" * 6
    ),
    "code (identifiers)": (
        "user_account_settings_processor user_account_settings_processor_factory "
        "HTTPResponseWriterFactory configureLoggingLevel getRequestContextHolder "
        "AbstractConfigurableBeanDefinitionRegistrar " * 8
    ),
    "japanese": "これはバイト列で動作する言語モデルについての説明です。" * 20,
    "chinese": "这是一个关于在字节级别运行的语言模型的说明。" * 20,
    "arabic": "هذا وصف لنموذج لغوي يعمل على مستوى البايت." * 20,
    "devanagari": "यह बाइट स्तर पर काम करने वाले भाषा मॉडल का विवरण है।" * 20,
    "russian": "Это описание языковой модели, работающей на уровне байтов." * 20,
    "emoji": "🙂🎉🚀 the model sees raw bytes not tokens 🎯" * 12,
    "dna": "ACGTACGTACGGGGCTTAGCATCGATCGGCTAGCTAGCATCGATCGT" * 20,
    "protein": "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEENFKALVLIAFAQYL" * 8,
    "identifiers/urls": (
        "https://api.internal.example.com/v2/resources/12345/children?parent=99 "
        "org.apache.hadoop.hdfs.protocol.DatanodeID uuid 550e8400-e29b-41d4 "
    ) * 10,
}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-17m")
    table = build_token_byte_table(tokenizer)

    print(f"{'domain':22} {'bytes/token':>11} {'tokens/MB':>10}")
    print("-" * 47)
    rows = []
    for name, text in SAMPLES.items():
        a = align(tokenizer(text, add_special_tokens=False)["input_ids"], table)
        ratio = a.bytes_per_patch
        rows.append((name, ratio))
        tokens_per_mb = 1_000_000 / ratio
        print(f"{name:22} {ratio:11.2f} {tokens_per_mb:10.0f}")

    english = dict(rows)["english prose"]
    print()
    print(f"english prose baseline: {english:.2f} bytes/patch (Bolmo's target is ~4.3)")
    worst = sorted(rows, key=lambda r: r[1])[:3]
    best = sorted(rows, key=lambda r: r[1], reverse=True)[:3]
    print(f"most compressible: {', '.join(f'{n} {r:.2f}' for n, r in best)}")
    print(f"least compressible: {', '.join(f'{n} {r:.2f}' for n, r in worst)}")
    ratios = [r for _, r in rows]
    print(
        f"spread: {min(ratios):.2f}-{max(ratios):.2f} "
        f"({max(ratios) / min(ratios):.1f}x), median {statistics.median(ratios):.2f}"
    )


if __name__ == "__main__":
    main()
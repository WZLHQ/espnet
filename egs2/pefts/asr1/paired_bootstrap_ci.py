import numpy as np
from jiwer import process_words


def read_trn(path, prefix=""):
    data = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            idx = line.rfind("(")

            text = line[:idx].strip()
            utt = line[idx + 1:-1].strip()

            if prefix:
                utt = prefix + "_" + utt

            data[utt] = text

    return data


def prepare_statistics(refs, hyps):
    """
    每句话只计算一次编辑距离。

    返回

    errors
    ref_words
    """

    utts = list(refs.keys())

    errors = np.zeros(len(utts), dtype=np.int32)
    ref_words = np.zeros(len(utts), dtype=np.int32)

    for i, utt in enumerate(utts):

        out = process_words(
            refs[utt],
            hyps[utt],
        )

        errors[i] = (
            out.substitutions
            + out.deletions
            + out.insertions
        )

        ref_words[i] = out.hits + out.substitutions + out.deletions

    return utts, errors, ref_words


def bootstrap_ci(errors,
                 ref_words,
                 n_boot=10000,
                 seed=42):

    rng = np.random.default_rng(seed)

    N = len(errors)

    wers = np.empty(n_boot)

    for i in range(n_boot):

        idx = rng.integers(
            0,
            N,
            N,
        )

        total_err = errors[idx].sum()

        total_ref = ref_words[idx].sum()

        wers[i] = 100 * total_err / total_ref

    overall = 100 * errors.sum() / ref_words.sum()

    low = np.percentile(wers, 2.5)

    high = np.percentile(wers, 97.5)

    return overall, low, high

def compare_two_systems(
    name1,
    overall1,
    low1,
    high1,
    name2,
    overall2,
    low2,
    high2,
):
    """
    Compare two ASR systems.

    Parameters
    ----------
    name1 : str
    overall1 : float
    low1 : float
    high1 : float

    name2 : str
    overall2 : float
    low2 : float
    high2 : float
    """

    diff = overall1 - overall2

    overlap = not (high1 < low2 or high2 < low1)

    print("=" * 60)
    print(f"{'Method':<15}{'WER':>10}{'95% CI':>25}")
    print("-" * 60)

    print(
        f"{name1:<15}"
        f"{overall1:>9.1f}%"
        f"{f'[{low1:.1f}, {high1:.1f}]':>25}"
    )

    print(
        f"{name2:<15}"
        f"{overall2:>9.1f}%"
        f"{f'[{low2:.1f}, {high2:.1f}]':>25}"
    )

    print("-" * 60)
    print(f"Difference ({name1} - {name2}) : {diff:.1f}%")

    if overlap:
        print("95% confidence intervals overlap.")
    else:
        print("95% confidence intervals do NOT overlap.")

    print("=" * 60)


############################################################
# configuration
############################################################

methods = ["DictLoRA4VeLoRA","DictLoRA4PCAM"]
keys = ["R32_7best_E1","R32_7best_A1"]
overall_list,low_list,high_list=[],[],[]
for method, key in zip(methods,keys):
    accents = ["US","UK","IND","CHN","JPN","PT","RU","KR","CA","ES",]
    model_size = "base_en"
    part = "test"
    all_ref = {}
    all_hyp = {}

    for accent in accents:
        ref_path = (
            f"espnet_outputs/"
            f"{accent}_whisper_{method}_outputs/"
            f"asr_{method}_whisper-{model_size}_{accent}-{key}/"
            f"decode_asr_whisper_noctc_beam3_asr_model_valid.acc.ave/"
            f"{accent}_{part}/score_wer/ref.trn"
        )

        hyp_path = (
            f"espnet_outputs/"
            f"{accent}_whisper_{method}_outputs/"
            f"asr_{method}_whisper-{model_size}_{accent}-{key}/"
            f"decode_asr_whisper_noctc_beam3_asr_model_valid.acc.ave/"
            f"{accent}_{part}/score_wer/hyp.trn"
        )

        ref = read_trn(ref_path, prefix=accent)
        hyp = read_trn(hyp_path, prefix=accent)

        if set(ref.keys()) != set(hyp.keys()):
            raise RuntimeError(f"{accent}: ref/hyp utterance ids do not match.")

        all_ref.update(ref)
        all_hyp.update(hyp)

    print(f"{method}")

    print(f"Total utterances: {len(all_ref)}")

    utts, errors, ref_words = prepare_statistics(
        all_ref,
        all_hyp,
    )

    overall, low, high = bootstrap_ci(
        errors,
        ref_words,
    )

    print(f"Overall WER : {overall:.1f}%")
    print(f"95% CI      : [{low:.1f}, {high:.1f}]")

    overall_list.append(overall)
    low_list.append(low)
    high_list.append(high)

compare_two_systems(
    "VeLoRA",
    overall_list[0],
    low_list[0],
    high_list[0],
    "P-CAM",
    overall_list[1],
    low_list[1],
    high_list[1],
)



TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_TOKENS = 3000

DECODING_KEYS = ("temperature", "top_p", "top_k")


def sampling_params(**overrides):
    refused = [k for k in DECODING_KEYS if k in overrides]
    if refused:
        raise ValueError(
            f"{', '.join(refused)} cannot be set per call. This repo evaluates at "
            f"temperature {TEMPERATURE}, top_p {TOP_P}, top_k {TOP_K}; change the "
            "constants in eval/config.py if you mean to change the protocol."
        )
    from vllm import SamplingParams

    kwargs = dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS)
    kwargs.update(overrides)
    return SamplingParams(**kwargs)


def scoring_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1, prompt_logprobs=0
    )


def record():
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
    }

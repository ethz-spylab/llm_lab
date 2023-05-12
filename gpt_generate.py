import torch
import numpy as np

generate_kwargs = {
    "do_sample": True,
    "temperature": 0.75,
    "repetition_penalty": 2.0,
    "top_k": 0,
    "num_return_sequences": 1,
    "pad_token_id": 50256,
}


def hash_input_id(input_id: torch.LongTensor):
    """
    Hash the input id of the previous token to a PRNG seed.
    :param input_id: the id of the previous token
    :return: the seed
    """
    assert len(input_id.shape) == 1
    # very non-cryptographically secure
    ret = (input_id[-1].item() * 314159) % 0xDEADBEEF
    return ret


def gen_red_list(input_id: torch.LongTensor, vocab_size: int, frac_red: float = 0.5):
    """
    Generate a pseudorandom list of tokens to ban.
    :param input_id: The input id of the previous token.
    :param vocab_size: The size of the vocabulary.
    :param frac_red: The fraction of the vocabulary to red-list.
    :return: The list of tokens to red-list.
    """
    seed = hash_input_id(input_id)
    np.random.seed(seed)
    return np.random.choice(vocab_size, size=int(frac_red * vocab_size), replace=False)


def generate_with_seed(model, tokenizer, prompt, logits_processor=None,
                       min_new_tokens=100, max_new_tokens=100, seed=None):
    """
    Generate text from a prompt.
    :param model: The model to use for generation.
    :param tokenizer: The tokenizer to use for generation.
    :param prompt: The prompt to generate from.
    :param logits_processor: A processor to apply to the logits before generating each token.
    :param min_new_tokens: The minimum number of tokens to generate.
    :param max_new_tokens: The maximum number of tokens to generate.
    :param seed: The seed to use for generation.
    :return: The generated text, including the prompt.
    """

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    sample_outputs = model.generate(
        input_ids,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        logits_processor=[] if logits_processor is None else [logits_processor],
        **generate_kwargs
    )
    text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    return text

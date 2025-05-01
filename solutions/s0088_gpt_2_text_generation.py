import numpy as np

np.random.seed(42)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(
    x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b


def ffn(x: np.ndarray, c_fc: dict, c_proj: dict) -> np.ndarray:
    return linear(gelu(linear(x, **c_fc)), **c_proj)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x: np.ndarray, c_attn: dict, c_proj: dict, n_head: int) -> np.ndarray:
    x = linear(x, **c_attn)
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1))
    )
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x


def transformer_block(
    x: np.ndarray, mlp: dict, attn: dict, ln_1: dict, ln_2: dict, n_head: int
) -> np.ndarray:
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(
    inputs: list[int],
    wte: np.ndarray,
    wpe: np.ndarray,
    blocks: list[dict],
    ln_f: dict,
    n_head: int,
) -> np.ndarray:
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


def generate(
    inputs: list[int], params: dict, n_head: int, n_tokens_to_generate: int
) -> list[int]:
    for _ in range(n_tokens_to_generate):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate :]


def gen_text(prompt: str, n_tokens_to_generate: int = 40) -> str:
    np.random.seed(42)
    encoder, hparams, params = load_encoder_hparams_and_params()
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text


def load_encoder_hparams_and_params(
    model_size: str = "124M", models_dir: str = "models"
):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str) -> list[int]:
            tokens = text.strip().split()
            return [
                self.encoder_dict.get(token, self.encoder_dict["<UNK>"])
                for token in tokens
            ]

        def decode(self, token_ids: list[int]) -> str:
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join(
                [reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids]
            )

    hparams = {"n_ctx": 1024, "n_head": 12}

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [],
        "ln_f": {
            "g": np.ones(10),
            "b": np.zeros(10),
        },
    }

    encoder = DummyBPE()
    return encoder, hparams, params

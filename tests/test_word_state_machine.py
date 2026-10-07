import unittest

import torch

try:
    from vllm.v1.sample.tandem_sampler import TandemSampler
    IMPORT_ERROR = None
except Exception as exc:
    TandemSampler = None
    IMPORT_ERROR = exc


class ScriptedCoin:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, _metadata, _index, _prob):
        value = self.values[self.calls % len(self.values)]
        self.calls += 1
        return value


class FakeMetadata:
    def __init__(self):
        self.generators = {}
        self.output_token_ids = []


def make_sampler(boundary_ids=(), max_gap=32, coin_values=(True, False)):
    sampler = TandemSampler.__new__(TandemSampler)
    sampler.strategy = "word"
    sampler.prob_primary = 0.5
    sampler.chunk_size = 1
    sampler.max_gap_tokens = max_gap
    sampler._step = 0
    sampler._word_state = {}
    sampler.boundary_token_ids = frozenset(boundary_ids)
    coin = ScriptedCoin(coin_values)
    sampler._draw_active = coin
    return sampler, coin


def run_incremental(sampler, tokens, key="req0"):
    metadata = FakeMetadata()
    sampler._word_state[key] = sampler._word_replay([], metadata, 0)
    for token in tokens:
        sampler._word_post_update([key], [token], [True], metadata)
    return sampler._word_state[key]


@unittest.skipIf(TandemSampler is None, f"vllm not importable: {IMPORT_ERROR}")
class WordSchedule(unittest.TestCase):
    def test_no_redraw_without_a_boundary_or_gap(self):
        sampler, coin = make_sampler(boundary_ids={100}, max_gap=32,
                                     coin_values=[True, False])
        state = run_incremental(sampler, [1, 2, 3, 4, 5])
        self.assertEqual(coin.calls, 1, "only the initial draw should have run")
        self.assertIs(state[0], True)
        self.assertEqual(state[1], 5, "counter advances on every token")

    def test_redraw_at_a_boundary_token(self):
        sampler, coin = make_sampler(boundary_ids={100}, max_gap=32,
                                     coin_values=[True, False])
        state = run_incremental(sampler, [1, 2, 100, 3])
        self.assertEqual(coin.calls, 2)
        self.assertIs(state[0], False, "the boundary token handed over")
        self.assertEqual(state[1], 1, "counter restarted at the boundary")

    def test_redraw_when_max_gap_is_reached(self):
        sampler, coin = make_sampler(boundary_ids=set(), max_gap=4,
                                     coin_values=[True, False])
        state = run_incremental(sampler, [1, 2, 3, 4, 5])
        self.assertEqual(coin.calls, 2, "redraw on the 4th token, not the 5th")
        self.assertIs(state[0], False)
        self.assertEqual(state[1], 1)

    def test_gap_counter_restarts_after_a_boundary(self):
        sampler, coin = make_sampler(boundary_ids={7}, max_gap=3,
                                     coin_values=[True, False, True])
        state = run_incremental(sampler, [7, 1, 1, 1])
        self.assertEqual(coin.calls, 3, "one initial, one boundary, one gap")
        self.assertIs(state[0], True)
        self.assertEqual(state[1], 0)

    def test_no_state_means_no_update(self):
        sampler, coin = make_sampler(boundary_ids={100})
        metadata = FakeMetadata()
        sampler._word_post_update(["unknown"], [100], [True], metadata)
        self.assertEqual(coin.calls, 0)
        self.assertEqual(sampler._word_state, {})

    def test_replay_skips_async_placeholders(self):
        sampler, coin = make_sampler(boundary_ids=set(), max_gap=3,
                                     coin_values=[True, False])
        state = sampler._word_replay([1, -1, -1, 2], FakeMetadata(), 0)
        self.assertEqual(coin.calls, 1, "placeholders must not force a redraw")
        self.assertEqual(state[1], 2, "only the two real tokens counted")

    def test_replay_of_only_placeholders_is_a_fresh_state(self):
        sampler, coin = make_sampler(boundary_ids=set(), max_gap=3,
                                     coin_values=[True, False])
        state = sampler._word_replay([-1] * 10, FakeMetadata(), 0)
        self.assertEqual(coin.calls, 1)
        self.assertEqual(state[1], 0)

    def test_replay_reproduces_the_incremental_state(self):
        tokens = [1, 2, 7, 3, 4, 5, 6, 7, 8, 9]
        live, live_coin = make_sampler(boundary_ids={7}, max_gap=4,
                                       coin_values=[True, False, True, False])
        replayed, replay_coin = make_sampler(boundary_ids={7}, max_gap=4,
                                             coin_values=[True, False, True, False])
        self.assertEqual(run_incremental(live, tokens),
                         replayed._word_replay(tokens, FakeMetadata(), 0))
        self.assertEqual(live_coin.calls, replay_coin.calls)


@unittest.skipIf(TandemSampler is None, f"vllm not importable: {IMPORT_ERROR}")
class WordSelect(unittest.TestCase):
    def test_forward_selects_tokens_from_the_model_named_by_the_mask(self):
        from vllm.config import TandemConfig
        from vllm.v1.sample.metadata import SamplingMetadata
        from vllm.v1.sample.logits_processor import LogitsProcessors

        sampler = TandemSampler(TandemConfig(
            enabled=True, frozen_model="unused-test-model", selection_strategy="word",
            boundary_token_ids=[0, 1, 2]))
        sampler._draw_active = ScriptedCoin([True, False, False, True])
        metadata = SamplingMetadata(
            temperature=None, all_greedy=True, all_random=False,
            top_p=None, top_k=None, generators={}, max_num_logprobs=None,
            no_penalties=True, prompt_token_ids=None,
            frequency_penalties=torch.zeros(2), presence_penalties=torch.zeros(2),
            repetition_penalties=torch.ones(2), output_token_ids=[[], []],
            allowed_token_ids_mask=None, bad_words_token_ids={},
            logitsprocs=LogitsProcessors())
        # Distinct argmaxes make a silently unused junior immediately detectable.
        primary = torch.tensor([[9., 0., 0.], [9., 0., 0.]])
        junior = torch.tensor([[0., 9., 0.], [0., 0., 9.]])
        for expected_mask, expected_tokens in (([1, 0], [0, 2]), ([0, 1], [1, 0])):
            output, mask, _ = sampler(primary, junior, metadata,
                                      req_ids=["a", "b"], output_token_ids=[[], []])
            self.assertEqual(mask.tolist(), expected_mask)
            self.assertEqual(output.sampled_token_ids.flatten().tolist(), expected_tokens)

    def test_reports_the_stored_author_per_request(self):
        sampler, _ = make_sampler(boundary_ids={100}, coin_values=[True, False])
        chosen = sampler._word_select(2, torch.device("cpu"), FakeMetadata(),
                                      req_ids=["a", "b"],
                                      output_token_ids=[[], []])
        self.assertEqual(chosen.dtype, torch.bool)
        self.assertEqual(chosen.tolist(),
                         [sampler._word_state["a"][0],
                          sampler._word_state["b"][0]])

    def test_state_of_a_finished_request_is_dropped(self):
        sampler, _ = make_sampler(boundary_ids={100}, coin_values=[True])
        sampler._word_select(2, torch.device("cpu"), FakeMetadata(),
                             req_ids=["a", "b"], output_token_ids=[[], []])
        self.assertEqual(set(sampler._word_state), {"a", "b"})
        sampler._word_select(1, torch.device("cpu"), FakeMetadata(),
                             req_ids=["a"], output_token_ids=[[]])
        self.assertEqual(set(sampler._word_state), {"a"})


if __name__ == "__main__":
    unittest.main()

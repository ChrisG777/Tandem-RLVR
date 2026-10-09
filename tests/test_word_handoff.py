"""Validate native sampler evidence and interruption-safe word evaluation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
import common
import word_handoff


class WordHandoff(unittest.TestCase):
    def test_rejects_missing_authorship_and_illegal_switch(self):
        c = SimpleNamespace(token_ids=[1,2], tandem_model_mask=None,
                            text='x', finish_reason='stop')
        with self.assertRaisesRegex(ValueError, 'authorship'):
            word_handoff.record_completion(c, {9})
        c.tandem_model_mask = [1,0]
        with self.assertRaisesRegex(ValueError, 'switched outside'):
            word_handoff.record_completion(c, {9})
        word_handoff.record_completion(c, {1})

    def test_gap_backstop_allows_switch_after_32_tokens(self):
        c = SimpleNamespace(token_ids=[1]*33, tandem_model_mask=[1]*32+[0],
                            text='x', finish_reason='length')
        word_handoff.record_completion(c, {9})

    def test_resume_skips_saved_problems_and_rejects_changed_decoding(self):
        calls=[]
        def generate(prompts, params, **kwargs):
            calls.append(prompts)
            if len(calls)==2:
                raise RuntimeError('preemption')
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9,1],
                    tandem_model_mask=[1,0], text='1', finish_reason='stop')]) for _ in prompts]
        engine=SimpleNamespace(generate=generate, llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda:None)))
        tok=SimpleNamespace(get_vocab=lambda:{'Ġx':9}, encode=lambda p:[1])
        problems=[{'set':'aime24','idx':i,'content':str(i),'gt':'1'} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'train/assets/qwen3_word_boundary_ids.json'
            path.parent.mkdir(parents=True)
            path.write_text('[9]')
            with patch.object(common,'ROOT',root), patch.object(common,'load_problems',return_value=problems), \
                 patch.object(common,'chat_prefix',side_effect=lambda t,p:p), patch.object(common,'grade',return_value=1), \
                 patch.object(word_handoff,'build_engine',return_value=engine), \
                 patch.dict(sys.modules,{'vllm':SimpleNamespace(SamplingParams=lambda **kw:kw),
                  'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda p:tok))}):
                out=root/'word.json'
                with self.assertRaisesRegex(RuntimeError,'preemption'):
                    word_handoff.evaluate('senior','junior',out,n=1)
                self.assertEqual(len(json.loads(Path(str(out)+'.progress.json').read_text())['gens']),1)
                result=word_handoff.evaluate('senior','junior',out,n=1)
                self.assertEqual(calls[2],[str(i) for i in range(1,17)])
                self.assertEqual(calls[-1],['17','18','19'])
                self.assertEqual(result['senior_token_fraction'],.5)
                self.assertEqual(result['metrics']['macro']['pass@1'],1)
                with self.assertRaisesRegex(ValueError,'inputs differ'):
                    word_handoff.evaluate('senior','junior',out,n=1,decoding='evaluation')


if __name__ == '__main__':
    unittest.main()

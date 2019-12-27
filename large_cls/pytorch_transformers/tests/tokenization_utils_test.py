# coding=utf-8
# Copyright 2018 HuggingFace Inc..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import six

from pytorch_transformers import PreTrainedTokenizer
from pytorch_transformers.tokenization_gpt2 import GPT2Tokenizer

class TokenizerUtilsTest(unittest.TestCase):
    def check_tokenizer_from_pretrained(self, tokenizer_class):
        s3_models = list(tokenizer_class.max_model_input_sizes.keys())
        for model_name in s3_models[:1]:
            tokenizer = tokenizer_class.from_pretrained(model_name)
            self.assertIsNotNone(tokenizer)
            self.assertIsInstance(tokenizer, tokenizer_class)
            self.assertIsInstance(tokenizer, PreTrainedTokenizer)

            for special_tok in tokenizer.all_special_tokens:
                if six.PY2:
                    self.assertIsInstance(special_tok, unicode)
                else:
                    self.assertIsInstance(special_tok, str)
                special_tok_id = tokenizer.convert_tokens_to_ids(special_tok)
                self.assertIsInstance(special_tok_id, int)

    def test_pretrained_tokenizers(self):
        self.check_tokenizer_from_pretrained(GPT2Tokenizer)

if __name__ == "__main__":
    unittest.main()

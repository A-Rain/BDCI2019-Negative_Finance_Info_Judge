# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

from pytorch_transformers import (DistilBertConfig, DistilBertModel, DistilBertForMaskedLM,
                                  DistilBertForQuestionAnswering, DistilBertForSequenceClassification)

from .modeling_common_test import (CommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester


class DistilBertModelTest(CommonTestCases.CommonModelTester):

    all_model_classes = (DistilBertModel, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
                         DistilBertForSequenceClassification)
    test_pruning = True
    test_torchscript = True
    test_resize_embeddings = True
    test_head_masking = True

    class DistilBertModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=False,
                     use_labels=True,
                     vocab_size=99,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     intermediate_size=37,
                     hidden_act="gelu",
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     scope=None,
                    ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                choice_labels = ids_tensor([self.batch_size], self.num_choices)

            config = DistilBertConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                dim=self.hidden_size,
                n_layers=self.num_hidden_layers,
                n_heads=self.num_attention_heads,
                hidden_dim=self.intermediate_size,
                hidden_act=self.hidden_act,
                dropout=self.hidden_dropout_prob,
                attention_dropout=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                initializer_range=self.initializer_range)

            return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

        def check_loss_output(self, result):
            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])

        def create_and_check_distilbert_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = DistilBertModel(config=config)
            model.eval()
            (sequence_output,) = model(input_ids, input_mask)
            (sequence_output,) = model(input_ids)

            result = {
                "sequence_output": sequence_output,
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])

        def create_and_check_distilbert_for_masked_lm(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = DistilBertForMaskedLM(config=config)
            model.eval()
            loss, prediction_scores = model(input_ids, attention_mask=input_mask, masked_lm_labels=token_labels)
            result = {
                "loss": loss,
                "prediction_scores": prediction_scores,
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.check_loss_output(result)

        def create_and_check_distilbert_for_question_answering(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
            model = DistilBertForQuestionAnswering(config=config)
            model.eval()
            loss, start_logits, end_logits = model(input_ids, attention_mask=input_mask, start_positions=sequence_labels, end_positions=sequence_labels)
            result = {
                "loss": loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            self.parent.assertListEqual(
                list(result["start_logits"].size()),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["end_logits"].size()),
                [self.batch_size, self.seq_length])
            self.check_loss_output(result)

        def create_and_check_distilbert_for_sequence_classification(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
            config.num_labels = self.num_labels
            model = DistilBertForSequenceClassification(config)
            model.eval()
            loss, logits = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
            result = {
                "loss": loss,
                "logits": logits,
            }
            self.parent.assertListEqual(
                list(result["logits"].size()),
                [self.batch_size, self.num_labels])
            self.check_loss_output(result)

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids, 'attention_mask': input_mask}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = DistilBertModelTest.DistilBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DistilBertConfig, dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_distilbert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_sequence_classification(*config_and_inputs)

    # @pytest.mark.slow
    # def test_model_from_pretrained(self):
    #     cache_dir = "/tmp/pytorch_transformers_test/"
    #     for model_name in list(DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
    #         model = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir)
    #         shutil.rmtree(cache_dir)
    #         self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()

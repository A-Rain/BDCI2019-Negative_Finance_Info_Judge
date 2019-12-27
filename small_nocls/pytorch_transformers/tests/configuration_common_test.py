# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import copy
import os
import shutil
import json
import random
import uuid

import unittest
import logging


class ConfigTester(object):
    def __init__(self, parent, config_class=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.inputs_dict = kwargs

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, 'vocab_size'))
        self.parent.assertTrue(hasattr(config, 'hidden_size'))
        self.parent.assertTrue(hasattr(config, 'num_attention_heads'))
        self.parent.assertTrue(hasattr(config, 'num_hidden_layers'))

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)
        json_file_path = os.path.join(os.getcwd(), "config_" + str(uuid.uuid4()) + ".json")
        config_first.to_json_file(json_file_path)
        config_second = self.config_class.from_json_file(json_file_path)
        os.remove(json_file_path)
        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()

if __name__ == "__main__":
    unittest.main()
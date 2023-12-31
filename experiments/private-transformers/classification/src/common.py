# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

task_name2suffix_name = {"sst-2": "GLUE-SST-2", "mnli": "MNLI", "qqp": "QQP", "qnli": "QNLI"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
true_tags = ('y', 'yes', 't', 'true')

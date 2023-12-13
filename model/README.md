---
language:
  - en
  - zh
license: other
tasks:
  - text-generation
studios:
  - baichuan-inc/Baichuan-13B-Chatdemo
---

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<div align="center">
<h1>
  Baichuan 2
</h1>
</div>

<div align="center">
  <a href="https://github.com/baichuan-inc/Baichuan2" target="_blank">🦉GitHub</a> | 
  <a href="https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/file/view/master/wechat.jpeg" target="_blank">💬WeChat</a> | 
  <a href="https://modelscope.cn/studios/baichuan-inc/Baichuan-13B-Chatdemo/summary" target="_blank">🤖Demo</a> 
</div>
<div align="center">
🚀 <a href="https://www.baichuan-ai.com/" target="_blank">百川大模型在线对话平台</a> 已正式向公众开放 🎉
</div>

# 目录

- [📖 模型介绍](#模型介绍)
- [⚙️ 快速开始](#快速开始)
- [📊 Benchmark评估](#评估)
- [📜 声明与协议](#声明与协议)

# 模型介绍

- Baichuan 2 是[百川智能]推出的**新一代开源大语言模型**，采用 **2.6 万亿**  Tokens 的高质量语料训练。
- Baichuan 2 在多个权威的中文、英文和多语言的通用、领域 benchmark 上取得同尺寸**最佳**的效果。
- 本次发布包含有 **7B**、**13B** 的 **Base** 和 **Chat** 版本，并提供了 Chat 版本的 **4bits 量化**。
- 所有版本对学术研究完全开放。同时，开发者通过邮件申请并获得官方商用许可后，即可**免费商用**，请参考[协议](#协议)章节。
- 欢迎阅读我们的技术报告 [Baichuan 2: Open Large-scale Language Models] 获取更多信息。

本次发布版本和下载链接见下表：

|     |          基座模型        |          对齐模型        |       对齐模型 4bits 量化        |
|:---:|:--------------------:|:--------------------:|:--------------------------:|
| 7B  | [Baichuan2-7B-Base]  | [Baichuan2-7B-Chat]  | [Baichuan2-7B-Chat-4bits]  |
| 13B | [Baichuan2-13B-Base] | [Baichuan2-13B-Chat] | [Baichuan2-13B-Chat-4bits] |

# 快速开始

```python
import torch
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig
model_dir = snapshot_download("baichuan-inc/Baichuan2-13B-Chat", revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
model.generation_config = GenerationConfig.from_pretrained(model_dir)
messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)
messages.append({'role': 'assistant', 'content': response})
messages.append({"role": "user", "content": "背诵一下将进酒"})
response = model.chat(tokenizer, messages)
print(response)
```
在魔搭社区的免费算力上，也可以通过量化的方式使用13B对话模型
```python
import torch
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    False,
    True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True)
model_dir = snapshot_download("baichuan-inc/Baichuan2-13B-Chat", revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16,
                              quantization_config=quantization_config)
model.generation_config = GenerationConfig.from_pretrained(model_dir)
messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)
messages.append({'role': 'assistant', 'content': response})
messages.append({"role": "user", "content": "背诵一下将进酒"})
response = model.chat(tokenizer, messages)
print(response)
```
# Benchmark 结果

我们在[通用]、[法律]、[医疗]、[数学]、[代码]和[多语言翻译]六个领域的中英文权威数据集上对模型进行了广泛测试，更多详细测评结果可查看[GitHub]。

### 7B 模型结果

|                         | **C-Eval** | **MMLU** | **CMMLU** | **Gaokao** | **AGIEval** | **BBH** |
|:-----------------------:|:----------:|:--------:|:---------:|:----------:|:-----------:|:-------:|
|                         |  5-shot    |  5-shot  |  5-shot   | 5-shot     | 5-shot      | 3-shot  |
|        **GPT-4**        | 68.40      | 83.93    | 70.33     | 66.15      | 63.27       | 75.12   |
|    **GPT-3.5 Turbo**    | 51.10      | 68.54    | 54.06     | 47.07      | 46.13       | 61.59   |
|      **LLaMA-7B**       | 27.10      | 35.10    | 26.75     | 27.81      | 28.17       | 32.38   |
|      **LLaMA2-7B**      | 28.90      | 45.73    | 31.38     | 25.97      | 26.53       | 39.16   |
|       **MPT-7B**        | 27.15      | 27.93    | 26.00     | 26.54      | 24.83       | 35.20   |
|      **Falcon-7B**      | 24.23      | 26.03    | 25.66     | 24.24      | 24.10       | 28.77   |
|     **ChatGLM2-6B**     | 50.20      | 45.90    | 49.00     | 49.44      | 45.28       | 31.65   |
|    **[Baichuan-7B]**    | 42.80      | 42.30    | 44.02     | 36.34      | 34.44       | 32.48   |
| **[Baichuan2-7B-Base]** | 54.00      | 54.16    | 57.07     | 47.47      | 42.73       | 41.56   |

### 13B 模型结果

|                             | **C-Eval** | **MMLU** | **CMMLU** | **Gaokao** | **AGIEval** | **BBH** |
|:---------------------------:|:----------:|:--------:|:---------:|:----------:|:-----------:|:-------:|
|                             |  5-shot    |  5-shot  |  5-shot   | 5-shot     | 5-shot      | 3-shot  |
|          **GPT-4**          | 68.40      | 83.93    | 70.33     | 66.15      | 63.27       | 75.12   |
|      **GPT-3.5 Turbo**      | 51.10      | 68.54    | 54.06     | 47.07      | 46.13       | 61.59   |
|        **LLaMA-13B**        | 28.50      | 46.30    | 31.15     | 28.23      | 28.22       | 37.89   |
|       **LLaMA2-13B**        | 35.80      | 55.09    | 37.99     | 30.83      | 32.29       | 46.98   |
|       **Vicuna-13B**        | 32.80      | 52.00    | 36.28     | 30.11      | 31.55       | 43.04   |
| **Chinese-Alpaca-Plus-13B** | 38.80      | 43.90    | 33.43     | 34.78      | 35.46       | 28.94   |
|       **XVERSE-13B**        | 53.70      | 55.21    | 58.44     | 44.69      | 42.54       | 38.06   |
|  **[Baichuan-13B-Base]**    | 52.40      | 51.60    | 55.30     | 49.69      | 43.20       | 43.01   |
|  **[Baichuan2-13B-Base]**   | 58.10      | 59.17    | 61.97     | 54.33      | 48.17       | 48.78   |


## 训练过程模型

除了训练了 2.6 万亿 Tokens 的 [Baichuan2-7B-Base] 模型，我们还提供了在此之前的另外 11 个中间过程的模型（分别对应训练了约 0.2 ~ 2.4 万亿 Tokens）供社区研究使用（[训练过程checkpoint下载]）。下图给出了这些 checkpoints 在 C-Eval、MMLU、CMMLU 三个 benchmark 上的效果变化：

![checkpoint](https://modelscope.cn/api/v1/models/baichuan-inc/Baichuan2-7B-Base/repo?Revision=master&FilePath=media/checkpoints.jpeg&View=true)

# 声明与协议

## 声明

我们在此声明，我们的开发团队并未基于 Baichuan 2 模型开发任何应用，无论是在 iOS、Android、网页或任何其他平台。我们强烈呼吁所有使用者，不要利用
Baichuan 2 模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将 Baichuan 2
模型用于未经适当安全审查和备案的互联网服务。我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用
Baichuan 2 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

## 协议

* Baichuan 2 模型的社区使用需遵循[《Baichuan 2 模型社区许可协议》]。
* Baichuan 2 支持商用，如果将 Baichuan 2 模型或其衍生品用作商业用途，请您按照如下方式联系许可方，以进行登记并向许可方申请书面授权：联系邮箱 [opensource@baichuan-inc.com]。

[GitHub]:https://github.com/baichuan-inc/Baichuan2
[Baichuan2]:https://github.com/baichuan-inc/Baichuan2

[Baichuan-7B]:https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary
[Baichuan2-7B-Base]:https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary
[Baichuan2-7B-Chat]:https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary
[Baichuan2-7B-Chat-4bits]:https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-4bits/summary
[Baichuan-13B-Base]:https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary
[Baichuan2-13B-Base]:https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary
[Baichuan2-13B-Chat]:https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary
[Baichuan2-13B-Chat-4bits]:https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat-4bits/summary

[通用]:https://github.com/baichuan-inc/Baichuan2#%E9%80%9A%E7%94%A8%E9%A2%86%E5%9F%9F
[法律]:https://github.com/baichuan-inc/Baichuan2#%E6%B3%95%E5%BE%8B%E5%8C%BB%E7%96%97
[医疗]:https://github.com/baichuan-inc/Baichuan2#%E6%B3%95%E5%BE%8B%E5%8C%BB%E7%96%97
[数学]:https://github.com/baichuan-inc/Baichuan2#%E6%95%B0%E5%AD%A6%E4%BB%A3%E7%A0%81
[代码]:https://github.com/baichuan-inc/Baichuan2#%E6%95%B0%E5%AD%A6%E4%BB%A3%E7%A0%81
[多语言翻译]:https://github.com/baichuan-inc/Baichuan2#%E5%A4%9A%E8%AF%AD%E8%A8%80%E7%BF%BB%E8%AF%91

[《Baichuan 2 模型社区许可协议》]:https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/Baichuan2%20%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf

[邮件申请]: mailto:opensource@baichuan-inc.com
[Email]: mailto:opensource@baichuan-inc.com
[opensource@baichuan-inc.com]: mailto:opensource@baichuan-inc.com
[训练过程checkpoint下载]: https://huggingface.co/baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints
[百川智能]: https://www.baichuan-ai.com
[Baichuan 2: Open Large-scale Language Models]:https://cdn.baichuan-ai.com/paper/Baichuan2-technical-report.pdf
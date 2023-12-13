from typing import List
from queue import Queue

import torch


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    """
    构建输入
    """
    def _parse_messages(messages, split_role="user"):
        """
        解析消息
        """
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            # 如果第一条是系统消息, 解析出来
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            # 如果是用户消息, 且 round 存在, 则将 round 加入 rounds. 开始下一轮
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            # 将消息加入 round
            round.append(message)
        # 添加最后的 round
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    # 解析消息
    system, rounds = _parse_messages(messages, split_role="user")
    # 编码系统消息
    system_tokens = tokenizer.encode(system)
    # 最大历史消息长度
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    # 逆序
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            # 加一个用户 token 或者助手 token
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    # 构建输入 tokens
    input_tokens = system_tokens + history_tokens
    # 如果最后一条不是助手消息, 则加一个助手 token
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    # 截断, 保留最后的 max_input_tokens
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return torch.LongTensor([input_tokens]).to(model.device)


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(
                self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


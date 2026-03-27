# SFT 运行
如果想要运行 SFT 训练，可以参考以下步骤：
```
cd cs336_alignment
uv run sft.py
```
直接这么运行，不要 uv run python -m sft.py。

# 参考 handout
handout 为 cs336_sprint2025_assignment5_alignment.pdf，里面有关于 SFT 训练的详细说明和代码示例，可以参考其中的内容来理解和实现 SFT 训练。

# 其他要求
1. 不要防御型编程，只考虑目前环境中的版本，不要保留任何为了兼容其他版本而写的fallback。
2. 进行修改后自己进行简单的测试，确保修改后的代码是正确的
3. 运行需要GPU的代码前先查看所有显卡目前的占用，尽量选择没有被使用的显卡，使用CUDA_VISIBLE_DEVICES 来指定显卡。
4. 不要直接修改 ./tests 这个目录下的代码，这个目录是课程提供的
5. 本项目使用uv进行环境管理，使用 uv run 来运行代码。
6. 训练任务需要使用 screen 进行会话管理。需要保证我可以通过 screen -r 来进入 session，查看训练进度。
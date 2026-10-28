# 让 Claude 看图 · read-image 插件

适合：Claude Code 主模型不能看图（比如 DeepSeek）时，让 Claude 也能读图片。

原理：图片先转成 base64 发给智谱视觉模型，模型返回文字，Claude 再用这段文字回答你。

## 前提条件

- 已安装 Claude Code
- **Python 3.10 以上**（Windows 上命令是 `python`，macOS / Linux 上是 `python3`）
- 能访问 `open.bigmodel.cn`

> Windows 上如果终端里输入 `python` 没反应，说明它不在 PATH。装 Python 时勾选 "Add to PATH"，或把 `.mcp.json` 里的 `"command": "python"` 改成 Python 的完整路径（如 `C:\\Python311\\python.exe`）。

---

## macOS 用户注意（重要）

macOS 默认**没有 `python` 命令**，而且系统自带的 `python3` 往往还是 3.9，不满足 3.10+ 的要求。不处理这两点，MCP 服务会启动失败，工具根本不会出现。

### 1. 安装新版 Python

```bash
brew install python
python3 --version   # 确认 >= 3.10
```

### 2. 安装依赖

```bash
python3 -m pip install mcp pillow
```

### 3. 修改 .mcp.json

把这一行：

```json
"command": "python"
```

改成：

```json
"command": "python3"
```

### 4. 复制插件

```bash
cp -r read-image ~/.claude/skills/
```

后面的 Key 配置和验证步骤与 Windows 完全一致。

---

## 安装（3 步）

### 1. 安装依赖

```bash
pip install mcp pillow
```

macOS / Linux 如果 `pip` 不存在，用 `pip3 install mcp pillow`。

### 2. 复制插件

**Windows：**

```bat
xcopy /E /I read-image "%USERPROFILE%\.claude\skills\read-image"
```

**macOS / Linux：**

```bash
cp -r read-image ~/.claude/skills/
```

### 3. 填 API Key

去 https://open.bigmodel.cn/usercenter/apikeys 注册并创建 Key。

新建文件 `~/.claude/read-image.json`：

```json
{"api_key": "你的Key"}
```

Windows 完整路径：`C:\Users\你的用户名\.claude\read-image.json`

---

## 让 Claude 帮你配置（推荐）

装好后不用自己找配置文件，直接对 Claude 说：

```
帮我配置 read-image
```

它会先检查当前状态，然后一步步引导你：
1. 告诉你去哪里申请 Key
2. 请你把 Key 粘贴到对话里
3. 自动写入本机配置
4. 再验证一次是否可用

> 不想把 Key 粘进对话，也可以自己创建 `~/.claude/read-image.json`，见上一节。

---

## 验证

1. 重启 Claude Code
2. 输入 `/mcp`，看到 `vision` 显示 connected
3. 说一句：

```
帮我看看 C:\图片\test.png 里有什么
```

---

## 怎么用

直接说人话即可：

- 「读一下这张图：路径」
- 「把这张图里的文字提取出来」
- 「这张图表里第三个柱子是多少」
- 「这个界面是什么布局」

---

## 读不出来怎么办

在 Claude Code 里输入：

```
read_image_doctor
```

它会自动检查 Key、网络和可用模型，并告诉你问题在哪。

**macOS 上工具完全不出现？** 依次检查：

1. `.mcp.json` 里的 `command` 是否已改成 `python3`
2. `python3 --version` 是否 ≥ 3.10
3. `python3 -m pip show mcp` 是否装在了同一个 Python 下

---

## 可选

不想把 Key 存在磁盘上，可以改用环境变量：

**Windows：** `setx ZHIPU_API_KEY "你的Key"`
**macOS / Linux：** `export ZHIPU_API_KEY="你的Key"`

环境变量优先级高于配置文件。

---

## 文件说明

```
read-image/
├── SKILL.md                     # Skill 说明与引导流程
├── .claude-plugin/plugin.json   # 插件声明
├── .mcp.json                    # MCP 启动配置
├── requirements.txt             # Python 依赖
└── server/read_image_server.py  # 服务端实现
```

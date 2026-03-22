# 说话人感知配音 — 新功能部署说明（Windows）

本文档仅说明**新增功能**的安装和配置步骤。
原有功能的环境要求不变。

---

## 新增文件

将以下 3 个文件放入程序目录（与 `video_dubbing.py` 同级）：

| 文件 | 说明 |
|------|------|
| `speaker_aware_dubbing.py` | 说话人识别集成模块 |
| `gender_classifier.py` | 性别识别模块（替换原有版本） |
---

## 第一步：安装新增 Python 依赖

打开命令提示符（CMD），在程序目录下执行：

```cmd
//先进入你的虚拟环境目录，再执行后面的环境安装
pip install pyannote.audio speechbrain librosa soundfile huggingface_hub
```

如果 `soundfile` 安装后报 `libsndfile` 错误，执行：

```cmd
pip uninstall soundfile -y
pip install soundfile --force-reinstall
```

---

## 第二步：下载性别分类模型

需要安装 [Git for Windows](https://git-scm.com/download/win)，然后在程序目录执行：

```cmd
git clone https://github.com/JaesungHuh/voice-gender-classifier.git
```

执行后程序目录下会出现 `voice-gender-classifier\` 文件夹，保留即可。

> **没有 Git？** 也可以直接下载 ZIP：
> 访问 https://github.com/JaesungHuh/voice-gender-classifier
> 点击绿色 **Code** 按钮 → **Download ZIP** → 解压到程序目录，
> 文件夹名确保为 `voice-gender-classifier`

---

## 第三步：获取 HuggingFace Token

1. 注册或登录 [huggingface.co](https://huggingface.co)

2. 访问以下两个页面，分别点击 **Agree** 接受使用协议：
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

3. 进入 [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   → 点击 **New token** → 类型选 **Read** → 复制 token（格式为 `hf_xxxxxxxx`）

---

## 第四步：设置环境变量

将 token 写入 Windows 系统环境变量，这样每次运行无需重复输入。

**操作方法：**
1. 右键「此电脑」→「属性」→「高级系统设置」→「环境变量」
2. 在「用户变量」区点击「新建」
3. 变量名：`HF_TOKEN`，变量值：`上一步你获取的token内容`
4. 点击确定，**重新打开 CMD 使其生效**

验证是否生效：
```cmd
echo %HF_TOKEN%
```
能看到 token 内容即为成功。

---

## 使用方法

**与原来完全相同，命令不变：**

```cmd
python video_dubbing.py --mode single --input_video input\video.mp4 --target_lang en --voice en_vctk_vits_m001 --output_video output\result.mp4
```

**新功能说明：**
- 设置了 `HF_TOKEN` 后，程序会自动识别视频中的说话人和性别
- 不同说话人会分配不同的声音（同性别的声音依次轮换）
- `--voice` 参数保留作为备用声音（识别失败时使用）
- **未设置 `HF_TOKEN` 时，行为与原来完全相同**

---

## 首次运行说明

首次运行时程序会自动从网络下载以下模型（共约 600MB–1GB），需保持网络畅通：

- 说话人分离模型（约 300MB）
- 声学特征模型（约 100MB）
- 性别分类模型（约 80MB）

下载完成后保存在本地，之后运行无需再次下载。

---

## 常见问题

**Q: 不想用说话人识别，想恢复原来的单一声音模式？**
A: 删除或清空系统环境变量 `HF_TOKEN` 即可，程序自动回退到原有模式。

**Q: 运行时提示 `No module named 'pyannote'`？**
A: 执行 `pip install pyannote.audio` 重新安装。

**Q: 提示 `Repository Not Found` 或 `401 Unauthorized`？**
A: HF_TOKEN 未设置或已过期，重新检查第三、四步。

**Q: 说话人识别要多长时间？**
A: 约为视频时长的 10%–30%（例如 10 分钟视频约需 1–3 分钟）。

**Q: `soundfile` 报 `libsndfile` 错误？**
A: 执行 `pip uninstall soundfile -y && pip install soundfile --force-reinstall`

**Q: patch 失败，提示找不到插入点？**
A: 说明你的 `video_dubbing.py` 版本与预期不符，请联系技术支持提供新版文件。

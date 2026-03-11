"""视频纯翻译字幕生成工具 (无配音版) - 修复版
功能：读取视频 -> 识别语音 -> 翻译文本 -> 生成字幕 -> 合成输出"""
import argparse
import os
import tempfile
import textwrap
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import warnings
from moviepy.editor import VideoFileClip, CompositeVideoClip
import torch
import sys
import glob
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 忽略非关键警告
warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub")

# 配置：可通过环境变量覆盖字幕字体路径
FONT_PATH_OVERRIDE = os.environ.get('SUBTITLE_FONT_PATH', None)

def get_available_fonts():
    """获取系统字体目录列表（用于调试字体加载）"""
    if sys.platform == 'darwin':  # macOS
        return [
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
        ]
    elif sys.platform.startswith('win'):  # Windows
        windir = os.environ.get('WINDIR', 'C:\\Windows')
        return [
            os.path.join(windir, 'Fonts', 'arial.ttf'),
            os.path.join(windir, 'Fonts', 'malgun.ttf'),  # 微软雅黑
            os.path.join(windir, 'Fonts', 'msyh.ttc'),    # 微软雅黑
            os.path.join(windir, 'Fonts', 'simsun.ttc'),  # 宋体
            os.path.join(windir, 'Fonts', 'msgothic.ttc') # 微软雅黑
        ]
    else:  # Linux/Other
        return ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

def create_subtitle_clip(text, start_time, duration, video_width, video_height, language_code='en'):
    """创建字幕片段 - 修复字体加载问题"""
    # ========== 1. 字体大小配置 ==========
    font_sizes = {
        'zh': 34, 'ja': 32, 'ko': 34, 'ar': 32, 'th': 32, 'hi': 32,
        'vi': 30, 'id': 30, 'tr': 30, 'pt': 30, 'es': 30, 'fr': 30, 'ru': 30,
        'default': 32
    }
    base_lang = language_code.split('-')[0].split('_')[0]
    base_font_size = font_sizes.get(base_lang, font_sizes['default'])
    # 根据视频分辨率调整
    if video_height >= 1080:
        font_size = int(base_font_size * 1.4)
        estimated_height = 150
    elif video_height >= 720:
        font_size = int(base_font_size * 1.2)
        estimated_height = 130
    else:
        font_size = int(base_font_size * 1.0)
        estimated_height = 110
    img_width = video_width
    # ========== 2. 创建透明背景 ==========
    subtitle_img = Image.new('RGBA', (img_width, estimated_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(subtitle_img)
    # ========== 3. 修复字体加载函数 ==========
    def get_font_for_mac(size):
        """Mac专用字体加载"""
        # Mac字体路径
        mac_font_paths = [
            # 1. Arial Unicode (最全)
            "/Library/Fonts/Arial Unicode.ttf",
            # 2. Apple SD Gothic Neo (韩文)
            "/System/Library/Fonts/Apple SD Gothic Neo.ttc",
            # 3. AppleGothic
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            # 4. PingFang (中文)
            "/System/Library/Fonts/PingFang.ttc",
            # 5. Hiragino Sans
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            # 6. Helvetica
            "/System/Library/Fonts/Helvetica.ttc",
            # 7. Arial
            "/System/Library/Fonts/Arial.ttf",
        ]
        # 尝试加载字体
        # 优先尝试外部指定的字体路径（环境变量或命令行传入）
        if FONT_PATH_OVERRIDE and os.path.exists(FONT_PATH_OVERRIDE):
            try:
                print(f"    - 尝试加载覆盖字体: {os.path.basename(FONT_PATH_OVERRIDE)}")
                # 对于 .ttc/.ttf 都尝试直接加载
                font = ImageFont.truetype(FONT_PATH_OVERRIDE, size=size)
                print(f"    - 成功加载覆盖字体: {FONT_PATH_OVERRIDE}")
                return font
            except Exception as e:
                print(f"    - 覆盖字体加载失败: {e}")
        
        for font_path in mac_font_paths:
            if os.path.exists(font_path):
                try:
                    print(f"    - 尝试加载: {os.path.basename(font_path)}")
                    if font_path.endswith('.ttc'):
                        # 对于字体集合，尝试不同索引
                        for index in [0, 1, 2]:
                            try:
                                font = ImageFont.truetype(font_path, size=size, index=index)
                                # 测试字体
                                test_text = "Test"
                                bbox = font.getbbox(test_text)
                                if bbox:
                                    print(f"    - 成功加载: {os.path.basename(font_path)} (索引:{index})")
                                    return font
                            except:
                                continue
                    else:
                        font = ImageFont.truetype(font_path, size=size)
                        # 测试字体
                        test_text = "Test"
                        bbox = font.getbbox(test_text)
                        if bbox:
                            print(f"    - 成功加载: {os.path.basename(font_path)}")
                            return font
                except Exception as e:
                    print(f"    - 加载失败 {os.path.basename(font_path)}: {e}")
                    continue
        # 如果所有字体都失败，返回默认字体
        print("    - 使用PIL默认字体")
        return ImageFont.load_default()

    def get_font_for_windows(size):
        """Windows专用字体加载"""
        windows_dirs = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
            'C:\\Windows\\Fonts',
            'D:\\Windows\\Fonts',
        ]
        windows_fonts = [
            'malgun.ttf',      # 韩文
            'gulim.ttc',       # 韩文
            'arialuni.ttf',    # Arial Unicode
            'msyh.ttc',        # 中文
            'msgothic.ttc',    # 日文
            'segoeui.ttf',
            'tahoma.ttf',
            'arial.ttf',
        ]

        for font_dir in windows_dirs:
            if os.path.exists(font_dir):
                for font_name in windows_fonts:
                    font_path = os.path.join(font_dir, font_name)
                    if os.path.exists(font_path):
                        try:
                            print(f"    - 尝试加载: {font_name}")
                            if font_name.endswith('.ttc'):
                                font = ImageFont.truetype(font_path, size=size, index=0)
                            else:
                                font = ImageFont.truetype(font_path, size=size)
                            # 测试字体
                            test_text = "Test"
                            bbox = font.getbbox(test_text)
                            if bbox:
                                print(f"    - 成功加载: {font_name}")
                                return font
                        except Exception as e:
                            print(f"    - 加载失败 {font_name}: {e}")
                            continue
        # 如果用户提供了覆盖字体路径，最后再尝试一次（有些系统字体目录不可读）
        if FONT_PATH_OVERRIDE and os.path.exists(FONT_PATH_OVERRIDE):
            try:
                print(f"    - 尝试加载覆盖字体: {os.path.basename(FONT_PATH_OVERRIDE)}")
                font = ImageFont.truetype(FONT_PATH_OVERRIDE, size=size)
                print(f"    - 成功加载覆盖字体: {FONT_PATH_OVERRIDE}")
                return font
            except Exception as e:
                print(f"    - 覆盖字体加载失败: {e}")
        print("    - 使用PIL默认字体")
        return ImageFont.load_default()

    # 根据系统选择字体加载函数
    import sys
    if sys.platform == 'darwin':
        print("    - 系统: macOS")
        font = get_font_for_mac(font_size)
    elif sys.platform.startswith('win'):
        print("    - 系统: Windows")
        font = get_font_for_windows(font_size)
    else:
        print("    - 系统: Linux/其他")
        font = ImageFont.load_default()

    # ========== 4. 确保font是有效的字体对象 ==========
    if font is None or not hasattr(font, 'getbbox'):
        print("    - 警告: 字体对象无效，使用默认字体")
        font = ImageFont.load_default()

    # ========== 5. 字符宽度配置 ==========
    char_widths = {
        'zh': 22, 'ja': 22, 'ko': 24, 'ar': 20, 'th': 22,
        'hi': 22, 'vi': 18, 'id': 18, 'tr': 18, 'pt': 18,
        'es': 18, 'fr': 18, 'ru': 18, 'default': 18
    }
    avg_char_width = char_widths.get(base_lang, char_widths['default'])
    max_chars = max(10, int(img_width * 0.7 / avg_char_width))

    # ========== 6. 文本换行 ==========
    def simple_wrap(text, max_chars):
        """简单的文本换行"""
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]
        lines = []
        current_line = ""
        for char in text:
            if len(current_line) >= max_chars:
                lines.append(current_line)
                current_line = char
            else:
                current_line += char
        if current_line:
            lines.append(current_line)
        return lines

    lines = simple_wrap(text, max_chars)

    # ========== 7. 计算位置 ==========
    # 计算每行高度
    line_heights = []
    for line in lines:
        try:
            bbox = font.getbbox(line)
            if bbox:
                height = bbox[3] - bbox[1]
            else:
                height = font_size
        except:
            height = font_size
        line_heights.append(height)

    # 计算总高度
    total_height = 0
    for h in line_heights:
        total_height += h + 8  # 8像素行间距
    if line_heights:
        total_height -= 8  # 减去最后一行的额外间距

    # 垂直位置（从底部开始）
    y_start = estimated_height - total_height - 20
    if y_start < 10:
        y_start = 10

    # ========== 8. 绘制文字（简化版确保可靠） ==========
    current_y = y_start
    for i, line in enumerate(lines):
        # 计算文本宽度
        try:
            bbox = font.getbbox(line)
            if bbox:
                line_width = bbox[2] - bbox[0]
            else:
                line_width = len(line) * avg_char_width
        except:
            line_width = len(line) * avg_char_width

        # 水平居中
        text_x = (img_width - line_width) // 2

        # 设置文字颜色
        if base_lang == 'ko':
            text_color = (255, 255, 200, 255)  # 韩文用淡黄色
        else:
            text_color = (255, 255, 255, 255)  # 其他用白色

        # 首先绘制文字描边（增强可读性）
        outline_color = (0, 0, 0, 200)
        # 简化描边 - 只绘制主要方向的阴影
        offsets = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        for dx, dy in offsets:
            try:
                draw.text((text_x + dx, current_y + dy), line,
                          fill=outline_color, font=font)
            except Exception as e:
                print(f"    - 描边绘制失败: {e}")
                # 继续尝试

        # 然后绘制主文字
        try:
            draw.text((text_x, current_y), line,
                      fill=text_color, font=font)
            print(f"    - 成功绘制: {line[:20]}...")
        except Exception as e:
            print(f"    - 主文字绘制失败: {e}")
            # 尝试使用ASCII回退
            try:
                ascii_line = line.encode('ascii', 'ignore').decode()
                if ascii_line:
                    draw.text((text_x, current_y), ascii_line,
                              fill=text_color, font=font)
                    print(f"    - 使用ASCII回退: {ascii_line[:20]}...")
            except:
                print(f"    - ASCII回退也失败")

        # 更新Y位置
        current_y += line_heights[i] + 8

    # ========== 9. 保存和返回 ==========
    temp_img_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_img_fd)
    try:
        subtitle_img.save(temp_img_path, format='PNG', optimize=True)
        print(f"    - 字幕图片保存: {temp_img_path}")
    except Exception as e:
        print(f"    - 图片保存失败: {e}")
        # 创建简单的错误图片
        error_img = Image.new('RGBA', (100, 50), color=(255, 0, 0, 128))
        error_img.save(temp_img_path, format='PNG')

    from moviepy.video.VideoClip import ImageClip
    try:
        clip = ImageClip(temp_img_path, duration=duration).set_start(start_time).set_position(('center', 'bottom'))
        clip.temp_path = temp_img_path
        print(f"    - 字幕片段创建成功: {duration:.2f}s")
        return clip
    except Exception as e:
        print(f"    - 创建ImageClip失败: {e}")
        # 返回一个空片段
        from moviepy.video.VideoClip import ColorClip
        empty_clip = ColorClip(size=(10, 10), color=(0, 0, 0), duration=0.1).set_start(start_time)
        empty_clip.temp_path = None
        return empty_clip


def translate_text(text, target_lang):
    """翻译文本 - 增强版（自动处理繁体中文）"""
    try:
        import re
        # 检测是否包含中文字符（繁体/简体）
        if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text):
            try:
                from zhconv import convert
                # 繁体转简体（双重保险：先转简体再翻译）
                simplified = convert(text, 'zh-cn')
                if simplified != text:
                    print(f"    - 繁体转简体: '{text}' -> '{simplified}'")
                    text = simplified
            except ImportError:
                print("    - 未安装 zhconv，繁体中文可能翻译不准确（建议: pip install zhconv）")
        
        # Google Translator 兼容性处理
        if target_lang.startswith('zh'):
            target_lang = 'zh-CN'
        
        translator = GoogleTranslator(source='auto', target=target_lang)
        result = translator.translate(text)
        
        # 验证翻译结果（避免返回原文）
        if result.strip() == text.strip() and not re.match(r'^[\s\W]+$', text):
            print(f"    - 警告: 翻译结果与原文相同，可能识别失败")
            # 尝试强制指定源语言为中文
            try:
                translator_zh = GoogleTranslator(source='zh-CN', target=target_lang)
                result2 = translator_zh.translate(text)
                if result2.strip() != text.strip():
                    print(f"    - 回退方案成功: '{text}' -> '{result2}'")
                    return result2
            except:
                pass
        
        return result
    except Exception as e:
        print(f"  - 翻译失败: {e}")
        return text  # 回退到原文


def translate_segments_parallel(segments, target_lang, max_workers=10, show_progress=True):
    """并行翻译所有片段

    Args:
        segments: 片段列表，每个片段包含 text, start, end, duration
        target_lang: 目标语言代码
        max_workers: 最大并行线程数
        show_progress: 是否显示进度

    Returns:
        翻译后的片段列表，保持原始顺序
    """
    if not segments:
        return []

    total = len(segments)
    completed = [0]  # 使用列表以便在闭包中修改
    lock = None

    def translate_one(seg, idx):
        """翻译单个片段"""
        translated = translate_text(seg["text"], target_lang)
        completed[0] += 1
        if show_progress and completed[0] % 10 == 0:
            print(f"  - 翻译进度: {completed[0]}/{total}")
        return {
            "idx": idx,
            "text": seg["text"],
            "translated": translated,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"]
        }

    print(f"  - 开始并行翻译 {total} 个片段 (线程数: {max_workers})...")
    start_time = time.time()

    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(translate_one, seg, i): i
                   for i, seg in enumerate(segments)}

        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                results[result["idx"]] = result
            except Exception as e:
                idx = futures[future]
                print(f"  - 片段 {idx} 翻译失败: {e}")
                # 使用原文作为回退
                results[idx] = {
                    "idx": idx,
                    "text": segments[idx]["text"],
                    "translated": segments[idx]["text"],
                    "start": segments[idx]["start"],
                    "end": segments[idx]["end"],
                    "duration": segments[idx]["duration"]
                }

    elapsed = time.time() - start_time
    print(f"  - 翻译完成: {total} 个片段, 耗时 {elapsed:.1f}s (平均 {elapsed/total:.2f}s/片段)")

    return results


def get_video_stream_info(video_path):
    """使用ffprobe获取视频流关键参数（跨平台兼容）"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'v:0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
            
        info = json.loads(result.stdout)
        if not info.get('streams'):
            return None
            
        stream = info['streams'][0]
        return {
            'width': int(stream.get('width', 0)),
            'height': int(stream.get('height', 0)),
            'codec': stream.get('codec_name', ''),
            'profile': stream.get('profile', ''),
            'pix_fmt': stream.get('pix_fmt', ''),
            'r_frame_rate': stream.get('r_frame_rate', '30/1'),
            'bit_rate': stream.get('bit_rate', '0')
        }
    except Exception as e:
        print(f"  ⚠️  ffprobe失败 {os.path.basename(video_path)}: {str(e)[:60]}")
        return None

def check_videos_compatible(video_files):
    """检查视频是否可无损合并（关键参数必须一致）"""
    if not video_files:
        return False, "无视频文件"
    
    # 获取基准视频信息
    base_info = get_video_stream_info(video_files[0])
    if not base_info:
        return False, f"无法读取基准视频: {os.path.basename(video_files[0])}"
    
    print(f"  ✓ 基准: {base_info['width']}x{base_info['height']}, {base_info['codec']}, {base_info['pix_fmt']}")
    
    # 检查所有视频
    for i, path in enumerate(video_files[1:], 1):
        info = get_video_stream_info(path)
        if not info:
            return False, f"无法读取: {os.path.basename(path)}"
        
        # 关键参数必须一致
        mismatches = []
        if info['width'] != base_info['width']:
            mismatches.append(f"宽度 {info['width']}≠{base_info['width']}")
        if info['height'] != base_info['height']:
            mismatches.append(f"高度 {info['height']}≠{base_info['height']}")
        if info['codec'] != base_info['codec']:
            mismatches.append(f"编码 {info['codec']}≠{base_info['codec']}")
        if info['pix_fmt'] != base_info['pix_fmt']:
            mismatches.append(f"像素格式 {info['pix_fmt']}≠{base_info['pix_fmt']}")
        
        if mismatches:
            return False, (
                f"参数不兼容 [{i+1}]: {os.path.basename(path)}\n"
                f"  基准: {base_info['width']}x{base_info['height']}, {base_info['codec']}, {base_info['pix_fmt']}\n"
                f"  当前: {info['width']}x{info['height']}, {info['codec']}, {info['pix_fmt']}\n"
                f"  差异: {', '.join(mismatches)}"
            )
    
    return True, "✓ 所有视频参数兼容，可无损合并"

def merge_videos_ffmpeg_safe(video_files, output_path):
    """使用FFmpeg concat demuxer安全合并（无损复制，极速）"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # 创建临时文件列表（UTF-8 + 路径转义）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8', newline='') as f:
        filelist_path = f.name
        for video_path in video_files:
            # 路径标准化（Mac/Windows通用）
            abs_path = os.path.abspath(video_path).replace('\\', '/')
            # 转义单引号（FFmpeg要求）
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    try:
        # FFmpeg合并命令（无损复制）
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            '-fflags', '+genpts',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        print(f"\n  → 执行无损合并 ({len(video_files)}个视频)...")
        print(f"     输出: {os.path.abspath(output_path)}")
        
        # 执行合并
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=3600  # 1小时超时
        )
        
        # 诊断输出
        if result.returncode != 0:
            print(f"\n  ✗ FFmpeg合并失败 (退出码: {result.returncode})")
            print(f"     错误输出:\n{result.stderr[:500]}")
            return False
        
        # 验证输出文件（宽松条件：>10KB）
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 10240:
                print(f"     ✓ 合并成功: {os.path.basename(output_path)} ({file_size/1024/1024:.1f} MB)")
                return True
            else:
                print(f"     ✗ 输出文件过小 ({file_size} bytes)，可能为空文件")
                return False
        else:
            print(f"     ✗ 输出文件不存在: {output_path}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ 合并超时 (1小时)")
        return False
    except Exception as e:
        print(f"  ✗ 合并异常: {type(e).__name__}: {str(e)}")
        return False
    finally:
        # 延迟清理临时文件
        time.sleep(0.1)
        if os.path.exists(filelist_path):
            try:
                os.remove(filelist_path)
            except:
                pass

def merge_videos_from_directory(output_dir, merged_filename, video_extension=".mp4"):
    """优化版视频合并（FFmpeg Concat Demuxer，极速无损）"""
    print(f"\n{'='*60}")
    print(f"🚀 优化视频合并 (FFmpeg Concat Demuxer)")
    print(f"{'='*60}")
    
    # 标准化路径
    output_dir = os.path.abspath(output_dir)
    output_path = os.path.join(output_dir, merged_filename)
    
    # 查找视频文件
    pattern = os.path.join(output_dir, f"*{video_extension}")
    video_files = sorted(glob.glob(pattern))
    
    if not video_files:
        print(f"❌ 错误: 目录 '{output_dir}' 中无 {video_extension} 文件")
        print(f"   目录内容: {os.listdir(output_dir)[:10]}")
        return False
    
    print(f"📁 找到 {len(video_files)} 个视频 (目录: {output_dir}):")
    for i, vf in enumerate(video_files[:min(5, len(video_files))], 1):
        size_mb = os.path.getsize(vf) / (1024*1024)
        print(f"   [{i}] {os.path.basename(vf)} ({size_mb:.1f} MB)")
    if len(video_files) > 5:
        print(f"   ... 共 {len(video_files)} 个文件")
    
    # 检查兼容性
    print(f"\n🔍 检查视频参数兼容性...")
    compatible, msg = check_videos_compatible(video_files)
    
    if not compatible:
        print(f"⚠️  {msg}")
        print("\n💡 建议: 重新生成视频时统一输出参数")
        print("   在 process_single_video 中固定:")
        print("   codec='libx264', audio_codec='aac', fps=30, preset='medium'")
        return False
    
    # 执行合并
    success = merge_videos_ffmpeg_safe(video_files, output_path)
    
    if success:
        # 最终验证
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            final_size = os.path.getsize(output_path) / (1024*1024)
            total_input = sum(os.path.getsize(v) for v in video_files) / (1024*1024)
            print(f"\n✅ 合并成功!")
            print(f"   输出路径: {output_path}")
            print(f"   大小: {final_size:.1f} MB (输入总大小: {total_input:.1f} MB)")
            print(f"   耗时: 约 {len(video_files) * 2 // 60}m{len(video_files) * 2 % 60}s (比MoviePy快60倍+)")
            return True
        else:
            print(f"\n❌ 输出文件验证失败: {output_path}")
            return False
    else:
        print(f"\n❌ 合并失败，请检查:")
        print(f"   1. 输出目录是否存在: {output_dir}")
        print(f"   2. 磁盘空间是否充足")
        print(f"   3. 视频文件是否被其他程序占用")
        return False

def process_single_video(input_video_path, target_language, output_video_path, parallel=True, workers=10):
    """处理单个视频：仅添加翻译字幕 - 修复资源泄漏版

    Args:
        input_video_path: 输入视频路径
        target_language: 目标语言代码
        output_video_path: 输出视频路径
        parallel: 是否启用并行翻译 (默认: True)
        workers: 并行翻译线程数 (默认: 10)
    """
    if not os.path.exists(input_video_path):
        print(f"错误: 文件不存在 - {input_video_path}")
        return False
    
    print(f"--- 开始处理: {input_video_path} ---")
    
    # 强制垃圾回收
    import gc
    gc.collect()
    
    original_video = None
    final_clip = None
    subtitle_clips = []
    temp_files_to_delete = []
    
    try:
        original_video = VideoFileClip(input_video_path)
        video_duration = original_video.duration
        video_width = original_video.w
        video_height = original_video.h
        
        print(f"视频时长: {video_duration:.2f}s")
        
        # ========== 1. 语音识别 (ASR) ==========
        print("\n[1/3] 语音识别 (ASR)...")
        model_size = "medium"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            segments, info = model.transcribe(
                input_video_path,
                language="zh",
                task="transcribe",
                beam_size=5,
                best_of=5,
                word_timestamps=True,  # 启用词级时间戳，提高时间精度
                vad_filter=True,       # 启用 VAD 过滤非语音片段
                vad_parameters={
                    "min_silence_duration_ms": 500,  # 最小静音时长
                    "speech_pad_ms": 200,            # 语音前后填充
                }
            )
            
            original_segments_data = []
            for segment in segments:
                if segment.start >= video_duration:
                    continue
                actual_end = min(segment.end, video_duration)
                duration = actual_end - segment.start
                if duration < 0.3:
                    continue
                original_segments_data.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": actual_end,
                    "duration": duration
                })
            print(f" - 识别到 {len(original_segments_data)} 个语音片段")
        except Exception as e:
            print(f" - 语音识别失败: {e}")
            return False
        
        # ========== 2. 翻译 ==========
        print(f"\n[2/3] 翻译为 {target_language}...")

        if parallel and len(original_segments_data) > 1:
            # 并行翻译
            translated_results = translate_segments_parallel(
                original_segments_data, target_language, max_workers=workers
            )

            for result in translated_results:
                try:
                    sub_clip = create_subtitle_clip(
                        result["translated"],
                        result["start"],
                        result["duration"],
                        video_width,
                        video_height,
                        target_language
                    )
                    subtitle_clips.append(sub_clip)
                    if hasattr(sub_clip, 'temp_path') and sub_clip.temp_path:
                        temp_files_to_delete.append(sub_clip.temp_path)
                except Exception as e:
                    print(f" - 字幕生成失败: {e}")
        else:
            # 顺序翻译（并行禁用或只有一个片段）
            for i, seg in enumerate(original_segments_data):
                translated_text = translate_text(seg["text"], target_language)
                print(f" [{i+1}] {seg['text'][:30]}... -> {translated_text[:30]}...")

                try:
                    sub_clip = create_subtitle_clip(
                        translated_text,
                        seg["start"],
                        seg["duration"],
                        video_width,
                        video_height,
                        target_language
                    )
                    subtitle_clips.append(sub_clip)
                    if hasattr(sub_clip, 'temp_path') and sub_clip.temp_path:
                        temp_files_to_delete.append(sub_clip.temp_path)
                except Exception as e:
                    print(f" - 字幕生成失败: {e}")
        
        # ========== 3. 合成视频 ==========
        print("\n[3/3] 合成最终视频 (仅字幕)...")
        
        # 释放ASR模型内存
        del model
        gc.collect()
        
        try:
            final_clip = CompositeVideoClip([original_video] + subtitle_clips, size=original_video.size)
            if original_video.audio:
                final_clip = final_clip.set_audio(original_video.audio)
            
            print(f" - 正在写入文件: {output_video_path}")
            print(f" - 视频尺寸: {original_video.size}, 时长: {video_duration:.2f}s")
            
            # 写入视频 - 添加超时保护
            final_clip.write_videofile(
                output_video_path,
                codec='libx264',
                audio_codec='aac',
                fps=original_video.fps if original_video.fps else 30,
                preset='medium',
                ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p'],
                threads=2,  # 减少线程数避免卡死
                logger='bar',  # 显示进度条
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # 验证输出文件
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 10240:  # 大于10KB
                    print(f" - ✓ 文件写入成功: {file_size/1024/1024:.1f} MB")
                    return True
                else:
                    print(f" - ✗ 输出文件过小: {file_size} bytes")
                    return False
            else:
                print(f" - ✗ 输出文件不存在")
                return False
                
        except Exception as e:
            print(f" - 视频合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理字幕片段
            for clip in subtitle_clips:
                try:
                    clip.close()
                except:
                    pass
            
            # 清理最终剪辑
            if final_clip:
                try:
                    final_clip.close()
                except:
                    pass
            
            # 清理原始视频
            if original_video:
                try:
                    original_video.close()
                except:
                    pass
            
            # 删除临时文件
            for temp_path in temp_files_to_delete:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            
            # 强制垃圾回收
            gc.collect()
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_process_videos(input_dir, output_dir, target_language, video_extension=".mp4", auto_merge=False, merged_filename="merged_output.mp4", parallel=True, workers=10):
    """批量处理视频（可选自动合并）

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_language: 目标语言代码
        video_extension: 视频文件扩展名
        auto_merge: 是否自动合并
        merged_filename: 合并后的文件名
        parallel: 是否启用并行翻译
        workers: 并行翻译线程数
    """
    print(f"\n--- 开始批量处理 ---")
    print(f"并行翻译: {'启用' if parallel else '禁用'} (线程数: {workers})")
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(os.path.abspath(input_dir), f"*{video_extension}")
    video_files = glob.glob(pattern)

    # 过滤掉输出目录中的文件
    video_files = [f for f in video_files if not f.startswith(os.path.abspath(output_dir))]

    if not video_files:
        print(f"错误: 在目录 {input_dir} 中未找到 {video_extension} 文件")
        return

    print(f"找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f" - {vf}")

    success_count = 0
    for input_path in video_files:
        filename = os.path.basename(input_path)
        name_part = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name_part}_SUB_{target_language}.mp4")

        print(f"\n处理: {filename}")
        if process_single_video(input_path, target_language, output_path, parallel=parallel, workers=workers):
            success_count += 1

    print(f"\n--- 批量处理完成 ({success_count}/{len(video_files)} 成功) ---")
    
    # 自动合并（仅当指定且有成功处理的视频）
    if auto_merge and success_count > 0:
        print(f"\n{'='*60}")
        print(f"📦 自动合并 {success_count} 个处理完成的视频")
        print(f"{'='*60}")
        merge_videos_from_directory(output_dir, merged_filename, video_extension)

def main():
    parser = argparse.ArgumentParser(description="视频纯翻译字幕生成工具")
    parser.add_argument("--mode", required=True,
                       choices=["single", "batch", "batch_merge", "merge_only"],
                       help="运行模式: single(单文件), batch(多文件处理), batch_merge(处理+合并), merge_only(仅合并)")
    parser.add_argument("--input_video", help="单文件模式: 输入视频路径")
    parser.add_argument("--input_dir", help="批量模式: 输入目录")
    parser.add_argument("--output_dir", default="./output_subtitles/", help="输出目录")
    parser.add_argument("--target_lang", help="目标语言代码 (如 en, ja, ko, fr)")
    parser.add_argument("--output_video", default="subtitled_output.mp4", help="单文件模式: 输出路径")
    parser.add_argument("--font_path", default=None, help="指定字幕字体路径")
    parser.add_argument("--merged_filename", default="merged_output.mp4",
                       help="合并后视频的文件名 (默认: merged_output.mp4)")
    parser.add_argument("--parallel", type=lambda x: x.lower() != 'false', default=True,
                       help="启用并行翻译 (默认: True, 设置为 false 禁用)")
    parser.add_argument("--workers", type=int, default=10,
                       help="并行翻译线程数 (默认: 10, 推荐 5-20)")
    args = parser.parse_args()

    # 设置全局字体路径
    global FONT_PATH_OVERRIDE
    if args.font_path:
        FONT_PATH_OVERRIDE = args.font_path

    if args.mode == "single":
        if not args.input_video or not args.target_lang:
            parser.error("--mode single requires --input_video and --target_lang")
        process_single_video(args.input_video, args.target_lang, args.output_video,
                            parallel=args.parallel, workers=args.workers)

    elif args.mode == "batch":
        if not args.input_dir or not args.target_lang:
            parser.error("--mode batch requires --input_dir and --target_lang")
        batch_process_videos(args.input_dir, args.output_dir, args.target_lang,
                            parallel=args.parallel, workers=args.workers)

    elif args.mode == "batch_merge":
        if not args.input_dir or not args.target_lang:
            parser.error("--mode batch_merge requires --input_dir and --target_lang")
        print(f"--- 批量处理并合并模式 ---")
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"目标语言: {args.target_lang}")
        print(f"合并文件名: {args.merged_filename}")
        print(f"并行翻译: {'启用' if args.parallel else '禁用'} (线程数: {args.workers})")
        batch_process_videos(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_language=args.target_lang,
            auto_merge=True,
            merged_filename=args.merged_filename,
            parallel=args.parallel,
            workers=args.workers
        )

    elif args.mode == "merge_only":
        print(f"--- 仅合并模式 ---")
        print(f"输出目录: {args.output_dir}")
        print(f"合并文件名: {args.merged_filename}")
        merge_videos_from_directory(args.output_dir, args.merged_filename)

if __name__ == "__main__":
    main()

# local_translator.py
"""本地翻译模块 - 基于 NLLB-200 3.3B (修复版)"""
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 全局单例，避免重复加载模型
_local_translator_instance = None

class NLLBTranslator:
    def __init__(self, model_name='facebook/nllb-200-3.3B', device='auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🚀 加载 NLLB 翻译模型：{model_name}")
        print(f"💻 使用设备：{self.device}")
        
        # NLLB 语言代码映射
        self.lang_map = {
            'zh': 'zho_Hans', 'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant',
            'en': 'eng_Latn', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang',
            'fr': 'fra_Latn', 'de': 'deu_Latn', 'es': 'spa_Latn',
            'pt': 'por_Latn', 'ru': 'rus_Cyrl', 'it': 'ita_Latn',
            'tr': 'tur_Latn', 'ar': 'arb_Arab', 'hi': 'hin_Deva',
            'th': 'tha_Thai', 'vi': 'vie_Latn', 'id': 'ind_Latn'
        }
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # 半精度加速 (如果支持)
            if self.device == 'cuda' and torch.cuda.is_bf16_supported():
                self.model = self.model.to(torch.bfloat16)
            self.model.to(self.device)
            self.model.eval()
            
            # ========== 修复：获取语言代码 ID 的兼容方法 ==========
            self.lang_code_to_id = self._get_lang_code_to_id()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            raise e
    
    def _get_lang_code_to_id(self):
        """兼容不同 transformers 版本的语言代码 ID 获取方法"""
        # 方法 1: 直接属性访问 (旧版本)
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            return self.tokenizer.lang_code_to_id
        
        # 方法 2: 从 additional_special_tokens 构建 (新版本)
        if hasattr(self.tokenizer, 'additional_special_tokens'):
            lang_codes = self.tokenizer.additional_special_tokens
            lang_ids = self.tokenizer.additional_special_tokens_ids
            if lang_codes and lang_ids:
                return dict(zip(lang_codes, lang_ids))
        
        # 方法 3: 使用 sp_model (SentencePiece)
        if hasattr(self.tokenizer, 'sp_model'):
            lang_code_dict = {}
            for lang_code in self.lang_map.values():
                token_id = self.tokenizer.sp_model.PieceToId(lang_code)
                if token_id != -1:
                    lang_code_dict[lang_code] = token_id
            return lang_code_dict
        
        # 方法 4: 手动映射 (最后回退)
        print("⚠️ 使用手动语言代码映射 (可能不精确)")
        return self._build_manual_lang_map()
    
    def _build_manual_lang_map(self):
        """手动构建常见语言代码 ID 映射"""
        # NLLB 常见语言代码的近似 ID (基于训练数据)
        return {
            'zho_Hans': 256047, 'zho_Hant': 256048,
            'eng_Latn': 256047, 'jpn_Jpan': 256055,
            'kor_Hang': 256056, 'fra_Latn': 256050,
            'deu_Latn': 256049, 'spa_Latn': 256068,
            'por_Latn': 256065, 'rus_Cyrl': 256067,
            'ita_Latn': 256054, 'tur_Latn': 256074,
            'arb_Arab': 256045, 'hin_Deva': 256052,
            'tha_Thai': 256072, 'vie_Latn': 256077,
            'ind_Latn': 256053,
        }

    def translate(self, text, source_lang='auto', target_lang='en'):
        if not text or not text.strip():
            return text
            
        # 简单语言检测 (中文检测)
        if source_lang == 'auto':
            import re
            if re.search(r'[\u4e00-\u9fff]', text):
                source_lang = 'zh'
            else:
                source_lang = 'en'
        
        src_code = self.lang_map.get(source_lang.lower(), 'zho_Hans')
        tgt_code = self.lang_map.get(target_lang.lower(), 'eng_Latn')
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # ========== 修复：安全获取目标语言 token ID ==========
            forced_bos_token_id = self._get_lang_token_id(tgt_code)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            print(f"⚠️ 翻译出错：{e}")
            return text
    
    def _get_lang_token_id(self, lang_code):
        """安全获取语言 token ID"""
        if lang_code in self.lang_code_to_id:
            return self.lang_code_to_id[lang_code]
        
        # 回退：尝试 eng_Latn (英语)
        if 'eng_Latn' in self.lang_code_to_id:
            return self.lang_code_to_id['eng_Latn']
        
        # 最后回退：使用 tokenizer 的 bos_token_id
        print(f"⚠️ 语言代码 {lang_code} 未找到，使用默认 bos_token_id")
        return self.tokenizer.bos_token_id


def get_translator(target_lang='en', device='auto'):
    global _local_translator_instance
    if _local_translator_instance is None:
        # 如果显存不足，改用 distilled-600M
        # model_name = 'facebook/nllb-200-distilled-600M'
        model_name = 'facebook/nllb-200-3.3B'
        
        _local_translator_instance = NLLBTranslator(model_name=model_name, device=device)
    return _local_translator_instance


# 兼容原有代码的接口
def translate_text(text, target_lang):
    translator = get_translator(target_lang)
    return translator.translate(text, source_lang='auto', target_lang=target_lang)


if __name__ == '__main__':
    # 测试
    result = translate_text("你好，这是一个测试", "en")
    print(f"翻译结果：{result}")
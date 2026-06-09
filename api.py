#!/usr/bin/env python
"""
VibeVoice ASR API Server

接收 base64 编码的音频，返回结构化转录结果（包含 start, end, speaker, content）。

启动方式:
    python api.py --model_path microsoft/VibeVoice-ASR --port 8000

或设置环境变量:
    VIBEVOICE_MODEL_PATH=microsoft/VibeVoice-ASR python api.py
"""

import os
import re
import sys
import argparse
import base64
import io
import tempfile
import time
import traceback
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from uvicorn import run as uvicorn_run
import requests

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# =============================================================================
# 请求/响应数据模型
# =============================================================================

class TranscribeRequest(BaseModel):
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64 编码的音频数据（支持 WAV、MP3、FLAC 等常见格式）。与 audio_url 二选一，优先使用此项。"
    )
    audio_url: Optional[str] = Field(
        default=None,
        description="音频文件 URL（支持 http/https）。与 audio_base64 二选一。"
    )
    context_info: Optional[str] = Field(
        default=None,
        description=(
            "可选的上下文信息，能显著提升专有名词和说话人识别准确率。"
            "推荐格式（可组合）："
            "'There are 2 speakers.'  固定说话人数量；"
            "'Speakers: Zhang San, Li Si.'  指定说话人姓名；"
            "'Topic: AI research, transformer architecture.'  限定领域术语；"
            "'Speaker 1 speaks Chinese, Speaker 2 speaks English.'  提示语种。"
        )
    )
    max_new_tokens: Optional[int] = Field(
        default=None, ge=1, le=32768,
        description="最大生成 token 数。留空（None）则根据音频时长自动估算，"
                    "推荐大多数场景使用自动估算以避免长音频截断。",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="采样温度，0 表示贪婪解码")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p 核采样阈值")
    num_beams: int = Field(
        default=1, ge=1, le=8,
        description=(
            "Beam search 束宽。1 = 贪婪解码（默认，最快）；"
            "推荐准确率优先场景设为 4（约 4x 慢，识别率通常提升 2-5%）。"
        )
    )
    repetition_penalty: float = Field(
        default=1.0, ge=1.0, le=2.0,
        description=(
            "重复惩罚系数。1.0 = 不惩罚（默认）；"
            "1.05-1.10 适合长音频或重复语音（防止'嗯...啊...'循环）；"
            "超过 1.2 会破坏正常重复（如'我我我'会被压成'我'）。"
        )
    )

    @model_validator(mode="after")
    def check_audio_source(self):
        if not self.audio_base64 and not self.audio_url:
            raise ValueError("必须提供 audio_base64 或 audio_url 之一")
        return self


class Segment(BaseModel):
    start: float = Field(..., description="片段开始时间（秒）")
    end: float = Field(..., description="片段结束时间（秒）")
    speaker: str = Field(..., description="说话人标识")
    content: str = Field(..., description="转录文本内容")


class TranscribeResponse(BaseModel):
    segments: List[Segment] = Field(..., description="结构化转录片段列表")
    raw_text: str = Field(..., description="模型原始输出文本")
    generation_time: float = Field(..., description="推理耗时（秒）")


# =============================================================================
# 工具函数
# =============================================================================

def estimate_max_new_tokens(duration_sec: float, upper_limit: int = 32768) -> int:
    """
    根据音频时长估算合适的 max_new_tokens。

    估算依据：
        - 输出文本密度约 2 token/s（含中英文常见情况）
        - 每段 JSON 序列化开销约 40 token（Start time / End time / Speaker ID / Content）
        - 假设平均段长 5s
        - 加 200 token buffer 防止边界

    Args:
        duration_sec: 音频时长（秒）
        upper_limit: 上限，默认 32768（与官方 demo 默认值一致）

    Returns:
        估算的 max_new_tokens 值，范围 [512, upper_limit]
    """
    text_tokens = int(duration_sec * 2)
    num_segments = max(1, int(duration_sec / 5))
    segment_overhead = num_segments * 40
    total = text_tokens + segment_overhead + 200
    return max(512, min(total, upper_limit))


# 匹配整段由中括号包裹的非语音事件：[Silence] / [Music] / [Laughter] /
# [Music] [Laughter] / [   ] 等
_BRACKETED_NOISE_PATTERN = re.compile(r'^\s*(?:\[[^\]]*\]\s*)+$')


def is_noise_segment(text: Optional[str]) -> bool:
    """
    判断转录片段是否仅包含中括号标注的非语音事件（噪声 / 静音 / 音乐等）。

    匹配示例（会被过滤）：
        - "[Silence]"
        - "[Music]"
        - "[Laughter]"
        - "[Music] [Laughter]"
        - "[  ]"

    不匹配（会保留）：
        - "Hello [Music] world"   ← 包含真实文字
        - "[2024-01-15] 开个会"   ← 包含真实文字
        - "你好"                  ← 无括号
        - "" 或 None              ← 视为噪声

    Args:
        text: 转录文本

    Returns:
        True 表示该片段是噪声，应被过滤
    """
    if not text:
        return True
    return bool(_BRACKETED_NOISE_PATTERN.match(text))


def probe_audio_duration_and_format(audio_bytes: bytes) -> tuple[Optional[float], str]:
    """
    探测音频时长和格式（不解码样本，避免 soundfile 重编码损失）。

    优先 soundfile.info（仅读 header，毫秒级），fallback 到 ffprobe
    （覆盖 MP3/M4A/AAC/AMR 等 soundfile 不识别的格式）。

    Returns:
        (duration_sec or None, file_suffix e.g. ".wav" / ".mp3" / ".bin")
    """
    # 1. soundfile 快速路径
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        return info.frames / info.samplerate, f".{info.format.lower()}"
    except Exception:
        pass

    # 2. ffprobe fallback（覆盖 ffmpeg 支持的所有格式）
    try:
        from subprocess import run
        result = run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration:format=format_name",
                "-of", "default=noprint_wrappers=1",
                "-",  # stdin
            ],
            input=audio_bytes,
            capture_output=True,
            check=True,
            timeout=10,
        )
        duration = None
        fmt_name = None
        for line in result.stdout.decode().strip().split("\n"):
            if line.startswith("duration="):
                try:
                    duration = float(line.split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("format_name="):
                # 可能有多个（如 "mov,mp4,m4a"），取第一个
                fmt_name = line.split("=", 1)[1].split(",")[0]
        suffix = f".{fmt_name}" if fmt_name else ".bin"
        return duration, suffix
    except Exception:
        pass

    return None, ".bin"


# =============================================================================
# ASR 推理封装
# =============================================================================

class VibeVoiceASRService:
    """ASR 服务封装，负责模型加载与单条音频推理。"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        attn_implementation: str = "auto",
    ):
        print(f"[ASR] Loading model from: {model_path}")

        # 自动推断设备与 attention 实现
        device, attn_implementation = self._resolve_device_and_attn(device, attn_implementation)
        print(f"[ASR] Device: {device}, Attention: {attn_implementation}")

        # 数据类型：CUDA 用 bfloat16，其余用 float32
        if device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # 加载 Processor
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        # 加载模型
        load_kwargs = {
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.device = device if device != "auto" else next(self.model.parameters()).device
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[ASR] Model loaded on {self.device}")
        print(f"[ASR] Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    def _resolve_device_and_attn(self, device: str, attn_implementation: str):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if attn_implementation == "auto":
            if device == "cuda":
                try:
                    import flash_attn  # noqa: F401
                    attn_implementation = "flash_attention_2"
                except ImportError:
                    print("[ASR] flash_attn not installed, falling back to sdpa")
                    attn_implementation = "sdpa"
            else:
                attn_implementation = "sdpa"
        return device, attn_implementation

    def transcribe(
        self,
        audio_path: str,
        context_info: Optional[str] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
    ) -> dict:
        """
        对单个音频文件进行转录。

        Args:
            audio_path: 音频文件路径（processor 内部会通过 ffmpeg/soundfile 加载并重采样到 24kHz）
            context_info: 可选上下文信息
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 阈值
            num_beams: Beam search 束宽
            repetition_penalty: 重复惩罚系数，1.0 表示不惩罚

        Returns:
            dict 包含 raw_text、segments、generation_time 等字段
        """
        # 1. 预处理
        inputs = self.processor(
            audio=audio_path,
            sampling_rate=None,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # 2. 构造生成配置
        do_sample = temperature > 0.0 and num_beams == 1
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p

        # 3. 推理
        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)
        generation_time = time.time() - start_time

        # 4. 解码输出
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # 5. 后处理结构化输出
        try:
            transcription_segments = self.processor.post_process_transcription(generated_text)
        except Exception as e:
            print(f"[ASR] Warning: Failed to parse structured output: {e}")
            transcription_segments = []

        return {
            "raw_text": generated_text,
            "segments": transcription_segments,
            "generation_time": generation_time,
        }


# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(
    title="VibeVoice ASR API",
    description="输入 base64 音频，输出带时间戳、说话人、内容的结构化转录结果。",
    version="1.0.0",
)

# 全局 ASR 服务实例（由 lifespan 或启动脚本初始化）
asr_service: Optional[VibeVoiceASRService] = None


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    接收 base64 编码音频，返回结构化 ASR 转录结果。
    """
    if asr_service is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded yet")

    # 打印入参（不打印 base64 原文，只打长度）
    print("[ASR] /transcribe request:")
    if request.audio_base64 is not None:
        print(f"  - audio_base64: <provided, len={len(request.audio_base64)} chars>")
    if request.audio_url is not None:
        print(f"  - audio_url: {request.audio_url}")
    print(f"  - context_info: {request.context_info!r}")
    print(
        f"  - max_new_tokens={request.max_new_tokens} (None=auto-size from duration), "
        f"temperature={request.temperature}, "
        f"top_p={request.top_p}, "
        f"num_beams={request.num_beams}, "
        f"repetition_penalty={request.repetition_penalty}"
    )

    # 准备音频 bytes 和来源标识
    audio_bytes: Optional[bytes] = None
    source_desc = ""

    if request.audio_base64:
        # 优先使用 base64
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
            source_desc = "audio_base64"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")
    elif request.audio_url:
        # 从 URL 下载
        try:
            resp = requests.get(request.audio_url, timeout=120)
            resp.raise_for_status()
            audio_bytes = resp.content
            source_desc = request.audio_url
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {e}")

    if audio_bytes is None:
        raise HTTPException(status_code=400, detail="No audio data available")

    print(f"[ASR] decoded audio_bytes: {len(audio_bytes)} bytes, source={source_desc}")

    # 1.5 提示：context_info 是提升准确率的最大单一杠杆
    if not request.context_info:
        print(
            "[ASR] 💡 hint: context_info not provided. For higher accuracy, "
            "pass hotwords / speaker names / domain terms via context_info "
            "(see field description for examples)."
        )

    # 2. 探测音频信息（不解码样本） + 写原始 bytes 临时文件（让 processor 用 ffmpeg 处理）
    #    避免 soundfile 重编码损失，同时正确处理 MP3/M4A/AAC/AMR 等格式
    audio_duration, audio_suffix = probe_audio_duration_and_format(audio_bytes)
    with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(audio_bytes)
    print(f"[ASR] temp file: {tmp_path}, size={len(audio_bytes)} bytes, "
          f"duration={audio_duration}s, suffix={audio_suffix}")

    # 2.5 音频质量诊断（帮助定位识别质量问题的原因）
    try:
        probe_buffer = io.BytesIO(audio_bytes)
        probe_array, probe_sr = sf.read(probe_buffer, dtype="float32")
        if probe_array.ndim > 1:
            probe_array = probe_array.mean(axis=1)
        peak = float(np.max(np.abs(probe_array)))
        rms = float(np.sqrt(np.mean(probe_array ** 2)))
        peak_db = 20 * np.log10(peak + 1e-10)
        rms_db = 20 * np.log10(rms + 1e-10)
        clipping = int(np.sum(np.abs(probe_array) > 0.99))
        print(f"[ASR] audio quality: sr={probe_sr}Hz, "
              f"peak={peak_db:.1f}dBFS, rms={rms_db:.1f}dBFS, "
              f"clipping_samples={clipping}")
        if peak_db > -1:
            print(f"[ASR] ⚠️  audio is clipping (peak={peak_db:.1f}dBFS > -1dBFS), "
                  f"speech content may be distorted")
        elif rms_db < -40:
            print(f"[ASR] ⚠️  audio is very quiet (rms={rms_db:.1f}dBFS < -40dBFS), "
                  f"consider amplifying before recognition")
    except Exception as e:
        # soundfile 不支持的格式（MP3/M4A/AAC 等）会失败，跳过即可
        print(f"[ASR] audio quality probe skipped: {e}")

    try:
        # 自适应 max_new_tokens：客户端未指定时按音频时长估算
        if request.max_new_tokens is None:
            if audio_duration is not None:
                resolved_max_new_tokens = estimate_max_new_tokens(audio_duration)
                print(f"[ASR] max_new_tokens auto-sized to {resolved_max_new_tokens} "
                      f"(audio_duration={audio_duration:.2f}s)")
            else:
                # sf.read 失败无法获取时长，使用保守默认值
                resolved_max_new_tokens = 16384
                print(f"[ASR] max_new_tokens fallback to {resolved_max_new_tokens} "
                      f"(audio_duration unknown)")
        else:
            resolved_max_new_tokens = request.max_new_tokens
            print(f"[ASR] max_new_tokens from request: {resolved_max_new_tokens}")

        # 3. 调用 ASR
        print(f"[ASR] calling transcribe on {tmp_path} ...")
        t0 = time.time()
        result = asr_service.transcribe(
            audio_path=tmp_path,
            context_info=request.context_info if request.context_info else None,
            max_new_tokens=resolved_max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
        )
        print(f"[ASR] transcribe done in {time.time() - t0:.2f}s, "
              f"raw_text_len={len(result.get('raw_text', ''))}, "
              f"segments={len(result.get('segments', []))}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        # 4. 清理临时文件
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # 5. 映射字段并返回（过滤纯括号噪声片段，如 [Silence] / [Music]）
    segments = []
    filtered_noise = 0
    for seg in result.get("segments", []):
        text = seg.get("text", "")
        if is_noise_segment(text):
            filtered_noise += 1
            continue
        segments.append(
            Segment(
                start=seg.get("start_time", 0.0),
                end=seg.get("end_time", 0.0),
                speaker=str(seg.get("speaker_id", "")),
                content=text,
            )
        )
    if filtered_noise:
        print(f"[ASR] filtered {filtered_noise} noise segment(s) (e.g. [Silence], [Music])")

    response = TranscribeResponse(
        segments=segments,
        raw_text=result["raw_text"],
        generation_time=result["generation_time"],
    )
    print(f"[ASR] returning {len(segments)} segments, raw_text_len={len(result['raw_text'])}")
    print(f"[ASR] response: {response.model_dump_json(indent=2, ensure_ascii=False)}")
    return response


@app.get("/health")
async def health():
    """健康检查接口。"""
    return {
        "status": "ok",
        "model_loaded": asr_service is not None,
    }


# =============================================================================
# 启动入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR API Server")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("VIBEVOICE_MODEL_PATH", "/root/models"),
        help="模型路径或 HuggingFace 模型名（也可通过环境变量 VIBEVOICE_MODEL_PATH 设置，默认：/root/models）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("VIBEVOICE_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu", "mps"],
        help="推理设备",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=os.environ.get("VIBEVOICE_ATTN", "auto"),
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention 实现方式",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("VIBEVOICE_HOST", "0.0.0.0"),
        help="监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VIBEVOICE_PORT", "6006")),
        help="监听端口（默认：6006）",
    )
    args = parser.parse_args()

    if not args.model_path:
        print("Error: --model_path is required (or set VIBEVOICE_MODEL_PATH env var)")
        sys.exit(1)

    # 加载模型（赋值全局变量）
    global asr_service
    asr_service = VibeVoiceASRService(
        model_path=args.model_path,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )

    print(f"\n[API] Starting server at http://{args.host}:{args.port}")
    print("[API] Endpoints:")
    print(f"  - POST http://{args.host}:{args.port}/transcribe")
    print(f"  - GET  http://{args.host}:{args.port}/health")
    print()

    uvicorn_run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

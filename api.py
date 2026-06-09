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
        description="可选的上下文信息（如热词、说话人姓名、主题等），帮助提升转写质量"
    )
    max_new_tokens: int = Field(default=8192, description="最大生成 token 数")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="采样温度，0 表示贪婪解码")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p 核采样阈值")
    num_beams: int = Field(default=1, ge=1, description="Beam search 束宽，1 表示贪婪解码")

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
        f"  - max_new_tokens={request.max_new_tokens}, "
        f"temperature={request.temperature}, "
        f"top_p={request.top_p}, "
        f"num_beams={request.num_beams}"
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

    # 2. Bytes -> 临时音频文件（让 processor 自动处理格式与重采样）
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        audio_array, sample_rate = sf.read(audio_buffer)
        # 如果 soundfile 能读，说明格式受支持，直接写成临时 WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_array, sample_rate)
        print(f"[ASR] temp wav: {tmp_path}, sr={sample_rate}, shape={getattr(audio_array, 'shape', None)}")
    except Exception as e:
        # soundfile 无法识别时，回退为直接写入原始 bytes（ffmpeg 可能支持）
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_bytes)
        print(f"[ASR] soundfile read failed ({e}); fallback to raw .bin: {tmp_path}")

    try:
        # 3. 调用 ASR
        print(f"[ASR] calling transcribe on {tmp_path} ...")
        t0 = time.time()
        result = asr_service.transcribe(
            audio_path=tmp_path,
            context_info=request.context_info if request.context_info else None,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_beams=request.num_beams,
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

    # 5. 映射字段并返回
    segments = []
    for seg in result.get("segments", []):
        segments.append(
            Segment(
                start=seg.get("start_time", 0.0),
                end=seg.get("end_time", 0.0),
                speaker=str(seg.get("speaker_id", "")),
                content=seg.get("text", ""),
            )
        )

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

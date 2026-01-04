from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Optional, Union

from google.cloud import speech


def _get_speech_client() -> speech.SpeechClient:
    """Get Speech client using service account JSON file."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS 환경변수에 서비스 계정 JSON 파일 경로를 설정해주세요."
        )
    if not Path(credentials_path).exists():
        raise FileNotFoundError(f"서비스 계정 파일을 찾을 수 없습니다: {credentials_path}")
    return speech.SpeechClient()


def transcribe_from_file(
    file_path: Union[str, Path],
    language_code: str = "ko-KR",
) -> Optional[str]:
    """
    Transcribes audio from a WAV file using Google Cloud Speech-to-Text.

    Args:
        file_path: Path to the WAV file
        language_code: BCP-47 language code (default: Korean)

    Returns:
        Transcribed text or None if no speech detected
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    print(f"[STT] 파일에서 음성을 읽습니다: {file_path}")

    # Read WAV file to get sample rate
    with wave.open(str(file_path), "rb") as wf:
        sample_rate_hz = wf.getframerate()
        channels = wf.getnchannels()
        print(f"[STT] 샘플레이트: {sample_rate_hz}Hz, 채널: {channels}")

    # Read audio content
    with open(file_path, "rb") as f:
        audio_content = f.read()

    print("[STT] Google STT로 변환 중...")

    client = _get_speech_client()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
        audio_channel_count=channels,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("[STT] 음성이 감지되지 않았습니다.")
        return None

    # Combine all results
    transcripts = [result.alternatives[0].transcript for result in response.results]
    transcript = " ".join(transcripts)
    print(f"[STT] 인식된 텍스트: {transcript}")
    return transcript


def transcribe_from_microphone(
    language_code: str = "ko-KR",
    sample_rate_hz: int = 16000,
    timeout_seconds: float = 10.0,
) -> Optional[str]:
    """
    Records audio from the microphone and transcribes it using Google Cloud Speech-to-Text.

    Args:
        language_code: BCP-47 language code (default: Korean)
        sample_rate_hz: Audio sample rate in Hz
        timeout_seconds: Maximum recording duration

    Returns:
        Transcribed text or None if no speech detected
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError("pyaudio is required. Install it with: pip install pyaudio")

    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    print(f"[STT] 마이크에서 음성을 녹음합니다... ({timeout_seconds}초)")

    # Record audio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate_hz,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    num_chunks = int(sample_rate_hz / CHUNK * timeout_seconds)

    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_content = b"".join(frames)
    print("[STT] 녹음 완료. Google STT로 변환 중...")

    client = _get_speech_client()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("[STT] 음성이 감지되지 않았습니다.")
        return None

    transcript = response.results[0].alternatives[0].transcript
    print(f"[STT] 인식된 텍스트: {transcript}")
    return transcript


def transcribe_streaming(
    language_code: str = "ko-KR",
    sample_rate_hz: int = 16000,
) -> Optional[str]:
    """
    Real-time streaming transcription from microphone using gRPC.
    Shows interim results as you speak.

    Args:
        language_code: BCP-47 language code (default: Korean)
        sample_rate_hz: Audio sample rate in Hz

    Returns:
        Final transcribed text or None if no speech detected
    """
    try:
        import pyaudio
    except ImportError:
        raise ImportError("pyaudio is required. Install it with: pip install pyaudio")

    import queue
    import threading

    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    client = _get_speech_client()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    audio_queue: queue.Queue = queue.Queue()
    stop_recording = threading.Event()

    def audio_generator():
        while not stop_recording.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate_hz,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("[STT] 실시간 음성 인식을 시작합니다. 말씀해 주세요... (Ctrl+C로 종료)")

    final_transcript = None

    def record_audio():
        while not stop_recording.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_queue.put(data)
            except Exception:
                break

    record_thread = threading.Thread(target=record_audio, daemon=True)
    record_thread.start()

    try:
        responses = client.streaming_recognize(streaming_config, audio_generator())

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript

            if result.is_final:
                print(f"\n[STT] 최종 인식: {transcript}")
                final_transcript = transcript
                break
            else:
                print(f"\r[STT] 인식 중: {transcript}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[STT] 음성 인식 종료")
    finally:
        stop_recording.set()
        audio_queue.put(None)
        stream.stop_stream()
        stream.close()
        p.terminate()

    return final_transcript


if __name__ == "__main__":
    # Test the STT functionality
    result = transcribe_from_file("test.wav")
    if result:
        print(f"Transcribed: {result}")
    else:
        print("No speech detected")

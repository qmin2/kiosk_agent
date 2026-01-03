
import time
import os
import sys
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # 02_Pseudo_Lab 폴더
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent  # .../kiosk_agent/src
AGENT_DIR = SRC_DIR.parent     # .../kiosk_agent
PROJECT_ROOT = AGENT_DIR.parent # .../02_Pseudo_Lab

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from kiosk_agent.src.models.gemini_client import GeminiClient
from kiosk_agent.src.config import ModelConfig

def main():
    models_to_test = [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview"
    ]

    screenshots_dir = AGENT_DIR / "screenshots"
    image_names = ["test1.jpeg", "test2.jpeg", "test3.jpeg", "test4.jpeg"]
    
    instruction = "햄버거를 주문할래"

    print(f"=== Gemini 모델 속도 비교 시작 ===")
    print(f"비교 모델: {models_to_test}")
    print(f"대상 이미지: {image_names}")
    print(f"이미지 경로: {screenshots_dir}\n")

    results = {}

    for model_name in models_to_test:
        print(f"Running test for model: [{model_name}]")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  Warning: GOOGLE_API_KEY environment variable is not set. Trying to read from config defaults if possible.")
        
        try:
            config = ModelConfig(
                provider="gemini",
                gemini_model=model_name,
                gemini_api_key=api_key 
            )
            
            client = GeminiClient(config=config)
        except Exception as e:
            print(f"  Failed to initialize client for {model_name}: {e}")
            continue

        model_times = []
        
        for img_name in image_names:
            img_path = screenshots_dir / img_name
            if not img_path.exists():
                print(f"  File not found: {img_name}")
                continue
            
            try:
                with Image.open(img_path) as image:
                    print(f"  > Processing {img_name} ... ", end="", flush=True)
                    start_time = time.time()
                    
                    dummy = client.generate(instruction, image)
                    
                    end_time = time.time()
                    elapsed = end_time - start_time
                    model_times.append(elapsed)
                    print(f"Done. ({elapsed:.4f}s)")
                    
                    time.sleep(1) 

            except Exception as e:
                print(f"Failed: {e}")
        
        if model_times:
            avg_time = sum(model_times) / len(model_times)
            results[model_name] = {
                "times": model_times,
                "avg": avg_time,
                "total": sum(model_times)
            }
            print(f"  => [{model_name}] Average: {avg_time:.4f}s\n")
        else:
            print(f"  => [{model_name}] No successful requests.\n")


    # 출력
    print("\n" + "="*50)
    print(f"{'Model Name':<25} | {'Average (s)':<12} | {'Total (s)':<12}")
    print("-" * 50)
    
    for model_name, data in results.items():
        print(f"{model_name:<25} | {data['avg']:<12.4f} | {data['total']:<12.4f}")
    print("="*50)

if __name__ == "__main__":
    main()

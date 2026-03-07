import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()
from openai import AsyncOpenAI
from pipeline import analyze_content, search_queue

client = AsyncOpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ.get('OPENROUTER_API_KEY')
)

async def test():
    await search_queue.start()
    
    async def progress(data):
        print(f"[PROGRESS] {data['message']}")
        
    print("Testing pipeline logic directly without Sanic HTTPX...")
    content = "Ini berita hoaks! Uang Rp 100 ribu di dalamnya ada microchip yang bisa melacak kita!"
    
    res = await analyze_content(client, content, emit_progress=progress)
    
    print("\n[FINAL RESULT]")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    
    await search_queue.stop()

if __name__ == "__main__":
    asyncio.run(test())

import httpx
import json
import asyncio


async def test():
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Hit the actual server we just started
        content = "Ini berita hoaks! Uang Rp 100 ribu di dalamnya ada microchip yang bisa melacak kita!"
        print(f"Testing point /api/analyze with content:\n{content}\n")

        async with client.stream(
            "POST", "http://127.0.0.1:8000/api/analyze", json={"content": content}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                        if payload["type"] == "progress":
                            print(f"[PROGRESS] {payload['data']['message']}")
                        elif payload["type"] == "result":
                            print("\n[FINAL RESULT]")
                            print(
                                json.dumps(
                                    payload["data"], indent=2, ensure_ascii=False
                                )
                            )
                        elif payload["type"] == "error":
                            print("\n[ERROR]")
                            print(payload["data"])
                    except json.JSONDecodeError:
                        pass


if __name__ == "__main__":
    asyncio.run(test())

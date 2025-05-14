import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport

async def main():
    transport = SSETransport(url="http://localhost:8000/sse")
    async with Client(transport) as client:
        print(await client.list_tools())
        print(await client.call_tool("configure_auto_labeling", {
            "selected_source": "ultralytics",
            "selected_model": "yolo11n"
        }))
        print(await client.call_tool("run_auto_labeling"))

if __name__ == "__main__":
    asyncio.run(main())

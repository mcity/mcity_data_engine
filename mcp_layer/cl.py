import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport

async def main():
    # Create an explicit SSE transport
    transport = SSETransport(url="http://localhost:8000/sse")

    async with Client(transport) as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        # Call the add tool
        result = await client.call_tool("add", {"a": 5, "b": 3})

        # Handle the result correctly based on its type
        if isinstance(result, list):
            print(f"Result: {result}")  # Print the list directly
        elif hasattr(result, 'text'):
            print(f"Result: {result.text}")  # Access text attribute if it exists
        else:
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")  # Print whatever the result is

            # If result is a complex object, try to inspect its attributes
            if hasattr(result, '__dict__'):
                print(f"Result attributes: {dir(result)}")

if __name__ == "__main__":
    asyncio.run(main())

from fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    # Explicitly use SSE transport on a specific host and port
    mcp.run(transport="sse", host="0.0.0.0", port=8000)

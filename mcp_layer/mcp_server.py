from mcptools import (
    mcp,  #shared FastMCP instance from __init__.py
    workflow_selector,
    auto_labeling,
    class_mapping,
    anomaly_detection,
    embedding_selection,
    zsal,
    ensemble_selection,
    data_ingest,
    v51,
    cvat_export,
    label_studio_export
)

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)

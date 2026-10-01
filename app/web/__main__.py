"""Run VOCA's private local web interface."""

import uvicorn


if __name__ == "__main__":
    print("VOCA is running locally.", flush=True)
    print("Open http://127.0.0.1:8000 in your browser.", flush=True)
    print("Press Ctrl+C here when you want to stop VOCA.", flush=True)
    uvicorn.run(
        "app.web.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="warning",
        access_log=False,
    )

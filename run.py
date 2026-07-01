import os

import uvicorn

from app import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("COMPANION_HOST", "0.0.0.0"),
        port=int(os.getenv("COMPANION_PORT", "8765")),
        reload=False,
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE"),
    )
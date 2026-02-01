"""WebSocket integration for streaming synthetic data generation."""

import json
from typing import Any

from genesis.streaming.generator import StreamingGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Default configuration constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_BATCH_SIZE = 100
DEFAULT_N_BATCHES = 10


class WebSocketStreamingGenerator:
    """Streaming generator over WebSocket for real-time applications.

    Provides a WebSocket interface for streaming synthetic data generation.
    Supports actions: generate, stream, stats.

    Example:
        >>> from genesis.streaming import StreamingGenerator, WebSocketStreamingGenerator
        >>>
        >>> stream = StreamingGenerator(method='gaussian_copula')
        >>> stream.fit(data)
        >>>
        >>> ws_server = WebSocketStreamingGenerator(stream, port=8765)
        >>> ws_server.run()  # Starts WebSocket server
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        """Initialize WebSocket streaming generator.

        Args:
            generator: Fitted streaming generator
            host: Host to bind to
            port: Port to bind to
        """
        self.generator = generator
        self.host = host
        self.port = port
        self._server = None

    async def handler(self, websocket: Any, path: str) -> None:
        """Handle WebSocket connection.

        Supports the following actions:
        - "generate": Generate a batch of samples
        - "stream": Stream multiple batches
        - "stats": Get generation statistics

        Args:
            websocket: WebSocket connection
            path: Request path (unused)
        """
        logger.info(f"New WebSocket connection from {websocket.remote_address}")

        try:
            async for message in websocket:
                request = json.loads(message)
                action = request.get("action", "generate")

                response = await self._handle_action(action, request)
                if response is not None:
                    await websocket.send(json.dumps(response))

                # Handle streaming separately as it sends multiple messages
                if action == "stream":
                    await self._handle_stream(websocket, request)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

    async def _handle_action(self, action: str, request: dict) -> dict | None:
        """Handle a single WebSocket action.

        Args:
            action: The action to perform
            request: The request data

        Returns:
            Response dict or None for streaming actions
        """
        if action == "generate":
            n_samples = request.get("n_samples", DEFAULT_BATCH_SIZE)
            conditions = request.get("conditions")

            synthetic = self.generator.generate(n_samples, conditions=conditions)

            return {
                "action": "data",
                "data": synthetic.to_dict(orient="records"),
                "n_samples": len(synthetic),
            }

        elif action == "stats":
            stats = self.generator.stats
            return {
                "action": "stats",
                "batches_generated": stats.batches_generated,
                "samples_generated": stats.samples_generated,
                "samples_per_second": stats.samples_per_second,
            }

        elif action == "stream":
            # Streaming is handled separately
            return None

        else:
            return {"error": f"Unknown action: {action}"}

    async def _handle_stream(self, websocket: Any, request: dict) -> None:
        """Handle streaming request.

        Args:
            websocket: WebSocket connection
            request: The request data
        """
        n_batches = request.get("n_batches", DEFAULT_N_BATCHES)
        batch_size = request.get("batch_size", DEFAULT_BATCH_SIZE)

        for batch in self.generator.generate_stream(n_batches, batch_size):
            response = {
                "action": "batch",
                "data": batch.to_dict(orient="records"),
                "n_samples": len(batch),
            }
            await websocket.send(json.dumps(response))

        await websocket.send(json.dumps({"action": "complete"}))

    def run(self) -> None:
        """Run the WebSocket server.

        This is a blocking call that runs the server until interrupted.
        """
        try:
            import asyncio

            import websockets
        except ImportError as e:
            raise ImportError(
                "websockets is required. Install with: pip install websockets"
            ) from e

        async def serve() -> None:
            async with websockets.serve(self.handler, self.host, self.port):
                logger.info(f"WebSocket server running at ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever

        asyncio.run(serve())

    async def run_async(self) -> None:
        """Run the WebSocket server asynchronously.

        Use this when running within an existing event loop.
        """
        try:
            import websockets
        except ImportError as e:
            raise ImportError(
                "websockets is required. Install with: pip install websockets"
            ) from e

        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"WebSocket server running at ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

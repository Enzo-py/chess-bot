
from src.utils.socket_server import ServerSocket
import src.utils.message as protocol

import asyncio

class Server:
    """
    Server class that handles the app.
    """

    def __init__(self):
        self.server = ServerSocket(_print=True)

    async def run(self):

        await self.server.start()

        self.server.on(
            ServerSocket.EVENTS_TYPES.on_client_connect,
            "client-connect",
            lambda client: asyncio.create_task(self.server.send(client, protocol.Message("confirm-connection", "Connection established").to_json()))
        )

        # Main loop
        while self.server.running:
            await self.server.wait_for_clients(1)
            await asyncio.sleep(2)

if __name__ == "__main__":
    Server = Server()
    asyncio.run(Server.run())


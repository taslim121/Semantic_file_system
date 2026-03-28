from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpTelemetry(ProductTelemetryClient):
    """Disable Chroma product telemetry completely."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return

"""Federated Generator Marketplace.

Decentralized marketplace for sharing trained generators (not data) across
organizations with cryptographic verification, model signing, and privacy-preserving
transfer protocols.

Features:
    - Federated discovery across nodes
    - Model signing and verification
    - ONNX export for portable generators
    - Privacy-preserving model transfer
    - Decentralized trust network
    - Token/credit economy
    - Model provenance chain

Example:
    Publish a generator to federated network::

        from genesis.federated_marketplace import (
            FederatedMarketplace,
            ModelPackage,
            NodeConfig,
        )

        # Initialize local node
        marketplace = FederatedMarketplace(
            node_config=NodeConfig(
                node_id="org_acme",
                signing_key="...",
                peers=["node1.genesis.io", "node2.genesis.io"],
            )
        )

        # Package and publish generator
        package = ModelPackage.from_generator(
            generator=my_fitted_generator,
            name="Customer Transaction Generator",
            description="Realistic e-commerce transactions",
            license="CC-BY-4.0",
        )
        
        listing_id = marketplace.publish(package)

    Discover and acquire generators::

        # Search across federated network
        results = marketplace.search(
            query="healthcare patient",
            filters={"license": "open", "min_quality": 0.9}
        )

        # Acquire generator
        generator = marketplace.acquire(
            listing_id=results[0].listing_id,
            payment_method="credits",
        )
        
        # Generate data locally
        synthetic_data = generator.generate(n_samples=10000)

Classes:
    NodeConfig: Federated node configuration.
    ModelPackage: Portable generator package.
    ModelSignature: Cryptographic model signature.
    FederatedListing: Listing in federated network.
    PeerNode: Remote peer node.
    TrustScore: Trust score for nodes.
    FederatedMarketplace: Main federated marketplace.
    DiscoveryProtocol: Decentralized discovery.
    TransferProtocol: Privacy-preserving transfer.
"""

import asyncio
import base64
import hashlib
import json
import pickle
import secrets
import struct
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class LicenseType(str, Enum):
    """Available license types for models."""

    MIT = "mit"
    APACHE_2 = "apache-2.0"
    CC_BY = "cc-by-4.0"
    CC_BY_NC = "cc-by-nc-4.0"
    COMMERCIAL = "commercial"
    CUSTOM = "custom"


class TransferStatus(str, Enum):
    """Model transfer status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationStatus(str, Enum):
    """Model verification status."""

    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class NodeConfig:
    """Configuration for a federated marketplace node.

    Attributes:
        node_id: Unique identifier for this node
        display_name: Human-readable node name
        signing_key: Private key for signing (base64)
        verify_key: Public key for verification (base64)
        peers: List of peer node addresses
        storage_path: Local storage directory
        max_model_size_mb: Maximum model size to accept
        accept_licenses: Licenses this node accepts
        credits_balance: Starting credit balance
        enable_relay: Whether to relay queries to peers
    """

    node_id: str
    display_name: str = ""
    signing_key: str = ""
    verify_key: str = ""
    peers: List[str] = field(default_factory=list)
    storage_path: str = "./federated_marketplace"
    max_model_size_mb: int = 500
    accept_licenses: List[LicenseType] = field(
        default_factory=lambda: [LicenseType.MIT, LicenseType.APACHE_2, LicenseType.CC_BY]
    )
    credits_balance: int = 100
    enable_relay: bool = True


@dataclass
class ModelSignature:
    """Cryptographic signature for a model package.

    Attributes:
        signer_id: ID of signing node
        signature: Base64-encoded signature
        algorithm: Signing algorithm used
        timestamp: When signature was created
        model_hash: Hash of model content
    """

    signer_id: str
    signature: str
    algorithm: str = "ed25519"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_hash: str = ""

    def verify(self, verify_key: bytes, model_bytes: bytes) -> bool:
        """Verify signature against model content.

        Args:
            verify_key: Public verification key
            model_bytes: Model content bytes

        Returns:
            True if signature is valid
        """
        try:
            # In production, use cryptography library
            # This is a simplified verification
            computed_hash = hashlib.sha256(model_bytes).hexdigest()
            return computed_hash == self.model_hash
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False


@dataclass
class ModelMetadata:
    """Metadata for a model package.

    Attributes:
        name: Model name
        description: Model description
        version: Model version
        generator_type: Type of generator (ctgan, tvae, etc.)
        schema: Schema of data this generates
        quality_score: Evaluated quality score
        privacy_score: Evaluated privacy score
        training_samples: Number of training samples used
        columns: List of columns
        domain: Domain (healthcare, finance, etc.)
        tags: Search tags
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    generator_type: str = "unknown"
    schema: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    privacy_score: float = 0.0
    training_samples: int = 0
    columns: List[str] = field(default_factory=list)
    domain: str = "general"
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelPackage:
    """Portable generator package for marketplace.

    Contains serialized generator, metadata, and signatures
    for federated distribution.
    """

    package_id: str
    metadata: ModelMetadata
    model_bytes: bytes  # Serialized generator
    format: str = "pickle"  # pickle, onnx, torchscript
    license: LicenseType = LicenseType.MIT
    signatures: List[ModelSignature] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    publisher_id: str = ""

    @classmethod
    def from_generator(
        cls,
        generator: Any,
        name: str,
        description: str = "",
        license: LicenseType = LicenseType.MIT,
        domain: str = "general",
        tags: Optional[List[str]] = None,
    ) -> "ModelPackage":
        """Create package from fitted generator.

        Args:
            generator: Fitted generator instance
            name: Model name
            description: Model description
            license: License type
            domain: Domain category
            tags: Search tags

        Returns:
            ModelPackage ready for publishing
        """
        # Serialize generator
        model_bytes = pickle.dumps(generator)

        # Extract metadata
        metadata = ModelMetadata(
            name=name,
            description=description,
            generator_type=type(generator).__name__,
            schema=generator.schema.to_dict() if hasattr(generator, "schema") and generator.schema else {},
            columns=list(generator.schema.columns.keys()) if hasattr(generator, "schema") and generator.schema else [],
            domain=domain,
            tags=tags or [],
        )

        return cls(
            package_id=str(uuid.uuid4()),
            metadata=metadata,
            model_bytes=model_bytes,
            license=license,
        )

    def to_generator(self) -> Any:
        """Extract generator from package.

        Returns:
            Deserialized generator
        """
        return pickle.loads(self.model_bytes)

    def compute_hash(self) -> str:
        """Compute hash of model content."""
        return hashlib.sha256(self.model_bytes).hexdigest()

    def size_bytes(self) -> int:
        """Get package size in bytes."""
        return len(self.model_bytes)

    def export_onnx(self, output_path: str) -> str:
        """Export generator to ONNX format.

        Args:
            output_path: Path for ONNX file

        Returns:
            Path to exported file

        Note:
            Only supported for certain generator types.
        """
        try:
            import torch

            generator = self.to_generator()

            # Check if generator has PyTorch model
            if hasattr(generator, "_model") and hasattr(generator._model, "forward"):
                # Export to ONNX
                dummy_input = torch.randn(1, 100)  # Adjust based on model
                torch.onnx.export(
                    generator._model,
                    dummy_input,
                    output_path,
                    input_names=["noise"],
                    output_names=["synthetic"],
                )
                return output_path

            raise ValueError("Generator does not support ONNX export")

        except ImportError:
            raise ImportError("ONNX export requires PyTorch: pip install torch")


@dataclass
class FederatedListing:
    """Listing in the federated marketplace.

    Attributes:
        listing_id: Unique listing ID
        package_id: ID of model package
        metadata: Model metadata
        publisher_node: Node that published this
        price_credits: Price in credits
        license: License type
        verification_status: Verification status
        trust_score: Publisher trust score
        download_count: Number of downloads
        rating: Average rating
        created_at: When listed
    """

    listing_id: str
    package_id: str
    metadata: ModelMetadata
    publisher_node: str
    price_credits: int = 0
    license: LicenseType = LicenseType.MIT
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    trust_score: float = 0.5
    download_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PeerNode:
    """Remote peer node in the federated network.

    Attributes:
        node_id: Peer's node ID
        address: Network address
        verify_key: Public key for verification
        trust_score: Local trust score for this peer
        last_seen: When peer was last contacted
        supported_protocols: Protocols this peer supports
    """

    node_id: str
    address: str
    verify_key: str = ""
    trust_score: float = 0.5
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    supported_protocols: List[str] = field(default_factory=lambda: ["http", "grpc"])
    is_online: bool = True
    latency_ms: float = 0.0


@dataclass
class SearchQuery:
    """Search query for federated discovery.

    Attributes:
        text: Text search query
        domain: Filter by domain
        license_types: Filter by licenses
        min_quality: Minimum quality score
        min_trust: Minimum trust score
        max_price: Maximum price in credits
        tags: Required tags
        limit: Maximum results
    """

    text: str = ""
    domain: Optional[str] = None
    license_types: List[LicenseType] = field(default_factory=list)
    min_quality: float = 0.0
    min_trust: float = 0.0
    max_price: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    limit: int = 50


@dataclass
class TransferRequest:
    """Request to transfer a model package."""

    request_id: str
    listing_id: str
    requester_node: str
    status: TransferStatus = TransferStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None


class DiscoveryProtocol:
    """Decentralized discovery protocol for federated network.

    Uses gossip-based propagation to discover listings
    across the federated network.
    """

    def __init__(self, local_node: "FederatedMarketplace"):
        """Initialize discovery protocol.

        Args:
            local_node: Local marketplace instance
        """
        self._local = local_node
        self._seen_queries: Set[str] = set()
        self._query_ttl = 5  # Max hops

    async def search(self, query: SearchQuery, peers: List[PeerNode]) -> List[FederatedListing]:
        """Search across federated network.

        Args:
            query: Search query
            peers: Peer nodes to query

        Returns:
            List of matching listings
        """
        results: List[FederatedListing] = []

        # Search local first
        local_results = self._search_local(query)
        results.extend(local_results)

        # Query peers in parallel
        if peers:
            tasks = [self._query_peer(peer, query) for peer in peers if peer.is_online]
            peer_results = await asyncio.gather(*tasks, return_exceptions=True)

            for peer_result in peer_results:
                if isinstance(peer_result, list):
                    results.extend(peer_result)

        # Deduplicate by listing_id
        seen = set()
        unique_results = []
        for listing in results:
            if listing.listing_id not in seen:
                seen.add(listing.listing_id)
                unique_results.append(listing)

        # Sort by relevance/trust
        unique_results.sort(key=lambda x: (x.trust_score, x.rating), reverse=True)

        return unique_results[: query.limit]

    def _search_local(self, query: SearchQuery) -> List[FederatedListing]:
        """Search local listings."""
        results = []

        for listing in self._local._listings.values():
            if self._matches_query(listing, query):
                results.append(listing)

        return results

    def _matches_query(self, listing: FederatedListing, query: SearchQuery) -> bool:
        """Check if listing matches query."""
        # Text search
        if query.text:
            text_lower = query.text.lower()
            if text_lower not in listing.metadata.name.lower() and text_lower not in listing.metadata.description.lower():
                return False

        # Domain filter
        if query.domain and listing.metadata.domain != query.domain:
            return False

        # License filter
        if query.license_types and listing.license not in query.license_types:
            return False

        # Quality filter
        if listing.metadata.quality_score < query.min_quality:
            return False

        # Trust filter
        if listing.trust_score < query.min_trust:
            return False

        # Price filter
        if query.max_price is not None and listing.price_credits > query.max_price:
            return False

        # Tags filter
        if query.tags:
            if not any(tag in listing.metadata.tags for tag in query.tags):
                return False

        return True

    async def _query_peer(self, peer: PeerNode, query: SearchQuery) -> List[FederatedListing]:
        """Query a peer node."""
        try:
            # In production, make HTTP/gRPC call to peer
            # This is a mock implementation
            logger.debug(f"Querying peer {peer.node_id} at {peer.address}")
            await asyncio.sleep(0.01)  # Simulate network latency
            return []  # Peer results would come from network
        except Exception as e:
            logger.warning(f"Failed to query peer {peer.node_id}: {e}")
            return []


class TransferProtocol:
    """Privacy-preserving model transfer protocol.

    Handles secure transfer of model packages between nodes
    with optional encryption and verification.
    """

    def __init__(self, local_node: "FederatedMarketplace"):
        """Initialize transfer protocol.

        Args:
            local_node: Local marketplace instance
        """
        self._local = local_node
        self._pending_transfers: Dict[str, TransferRequest] = {}

    async def request_transfer(
        self,
        listing_id: str,
        peer: PeerNode,
    ) -> TransferRequest:
        """Request model transfer from peer.

        Args:
            listing_id: Listing to transfer
            peer: Peer to request from

        Returns:
            TransferRequest
        """
        request = TransferRequest(
            request_id=str(uuid.uuid4()),
            listing_id=listing_id,
            requester_node=self._local.config.node_id,
        )

        self._pending_transfers[request.request_id] = request

        # In production, send request to peer
        # This is a mock implementation
        logger.info(f"Requesting transfer of {listing_id} from {peer.node_id}")

        return request

    async def complete_transfer(
        self,
        request: TransferRequest,
        package: ModelPackage,
    ) -> bool:
        """Complete a model transfer.

        Args:
            request: Transfer request
            package: Received model package

        Returns:
            True if transfer completed successfully
        """
        try:
            # Verify package
            if not self._verify_package(package):
                request.status = TransferStatus.FAILED
                request.error = "Package verification failed"
                return False

            # Store locally
            self._local._packages[package.package_id] = package

            request.status = TransferStatus.COMPLETED
            request.completed_at = datetime.utcnow().isoformat()

            logger.info(f"Transfer completed: {request.request_id}")
            return True

        except Exception as e:
            request.status = TransferStatus.FAILED
            request.error = str(e)
            return False

    def _verify_package(self, package: ModelPackage) -> bool:
        """Verify package integrity and signatures."""
        # Verify hash
        computed_hash = package.compute_hash()

        # Check signatures
        for sig in package.signatures:
            if sig.model_hash != computed_hash:
                return False

        return True


class CreditLedger:
    """Simple credit ledger for federated marketplace economy."""

    def __init__(self):
        """Initialize ledger."""
        self._balances: Dict[str, int] = {}
        self._transactions: List[Dict[str, Any]] = []

    def set_balance(self, node_id: str, balance: int) -> None:
        """Set node balance."""
        self._balances[node_id] = balance

    def get_balance(self, node_id: str) -> int:
        """Get node balance."""
        return self._balances.get(node_id, 0)

    def transfer(
        self,
        from_node: str,
        to_node: str,
        amount: int,
        reason: str = "",
    ) -> bool:
        """Transfer credits between nodes.

        Args:
            from_node: Source node
            to_node: Destination node
            amount: Amount to transfer
            reason: Transaction reason

        Returns:
            True if transfer succeeded
        """
        if self._balances.get(from_node, 0) < amount:
            return False

        self._balances[from_node] = self._balances.get(from_node, 0) - amount
        self._balances[to_node] = self._balances.get(to_node, 0) + amount

        self._transactions.append({
            "from": from_node,
            "to": to_node,
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return True


class FederatedMarketplace:
    """Federated marketplace for sharing trained generators.

    Enables decentralized discovery and transfer of generator
    models across organizations without sharing raw data.

    Example:
        >>> marketplace = FederatedMarketplace(NodeConfig(node_id="my_org"))
        >>> package = ModelPackage.from_generator(generator, "My Model")
        >>> listing_id = marketplace.publish(package)
        >>> results = await marketplace.search(SearchQuery(text="healthcare"))
    """

    def __init__(self, node_config: NodeConfig):
        """Initialize federated marketplace.

        Args:
            node_config: Configuration for this node
        """
        self.config = node_config
        self._packages: Dict[str, ModelPackage] = {}
        self._listings: Dict[str, FederatedListing] = {}
        self._peers: Dict[str, PeerNode] = {}
        self._discovery = DiscoveryProtocol(self)
        self._transfer = TransferProtocol(self)
        self._ledger = CreditLedger()

        # Initialize storage
        Path(node_config.storage_path).mkdir(parents=True, exist_ok=True)

        # Set initial balance
        self._ledger.set_balance(node_config.node_id, node_config.credits_balance)

        # Initialize signing keys if not provided
        if not node_config.signing_key:
            self._generate_keys()

        logger.info(f"Federated marketplace initialized: {node_config.node_id}")

    def _generate_keys(self) -> None:
        """Generate signing key pair."""
        # In production, use cryptography library
        # This is a simplified key generation
        self.config.signing_key = base64.b64encode(secrets.token_bytes(32)).decode()
        self.config.verify_key = base64.b64encode(secrets.token_bytes(32)).decode()

    def publish(
        self,
        package: ModelPackage,
        price_credits: int = 0,
    ) -> str:
        """Publish a model package to the marketplace.

        Args:
            package: Model package to publish
            price_credits: Price in credits (0 = free)

        Returns:
            Listing ID
        """
        # Sign package
        signature = self._sign_package(package)
        package.signatures.append(signature)
        package.publisher_id = self.config.node_id

        # Store package
        self._packages[package.package_id] = package

        # Create listing
        listing = FederatedListing(
            listing_id=str(uuid.uuid4()),
            package_id=package.package_id,
            metadata=package.metadata,
            publisher_node=self.config.node_id,
            price_credits=price_credits,
            license=package.license,
            verification_status=VerificationStatus.PENDING,
            trust_score=0.5,  # New listings start at neutral trust
        )

        self._listings[listing.listing_id] = listing

        # Persist to disk
        self._save_state()

        logger.info(f"Published listing: {listing.listing_id}")
        return listing.listing_id

    def _sign_package(self, package: ModelPackage) -> ModelSignature:
        """Sign a model package."""
        model_hash = package.compute_hash()

        # In production, use proper cryptographic signing
        # This is a simplified signature
        signature = base64.b64encode(
            hashlib.sha256(
                (self.config.signing_key + model_hash).encode()
            ).digest()
        ).decode()

        return ModelSignature(
            signer_id=self.config.node_id,
            signature=signature,
            model_hash=model_hash,
        )

    async def search(
        self,
        query: Union[str, SearchQuery],
        include_peers: bool = True,
    ) -> List[FederatedListing]:
        """Search for generators in the federated network.

        Args:
            query: Search query string or SearchQuery object
            include_peers: Whether to query peer nodes

        Returns:
            List of matching listings
        """
        if isinstance(query, str):
            query = SearchQuery(text=query)

        peers = list(self._peers.values()) if include_peers else []
        return await self._discovery.search(query, peers)

    def search_sync(
        self,
        query: Union[str, SearchQuery],
        include_peers: bool = False,
    ) -> List[FederatedListing]:
        """Synchronous search (local only by default).

        Args:
            query: Search query
            include_peers: Whether to include peers (requires async)

        Returns:
            List of matching listings
        """
        if isinstance(query, str):
            query = SearchQuery(text=query)

        # Local search only
        return self._discovery._search_local(query)

    async def acquire(
        self,
        listing_id: str,
        payment_method: str = "credits",
    ) -> Any:
        """Acquire a generator from the marketplace.

        Args:
            listing_id: Listing to acquire
            payment_method: Payment method (credits, free)

        Returns:
            Generator instance

        Raises:
            ValueError: If listing not found or insufficient funds
        """
        # Find listing
        listing = self._listings.get(listing_id)

        if not listing:
            # Try to find in peer network
            raise ValueError(f"Listing not found: {listing_id}")

        # Check if we own this listing (no payment needed)
        if listing.publisher_node == self.config.node_id:
            package = self._packages.get(listing.package_id)
            if package:
                return package.to_generator()

        # Check credits
        if payment_method == "credits" and listing.price_credits > 0:
            if self._ledger.get_balance(self.config.node_id) < listing.price_credits:
                raise ValueError("Insufficient credits")

            # Process payment
            self._ledger.transfer(
                self.config.node_id,
                listing.publisher_node,
                listing.price_credits,
                f"Acquired listing {listing_id}",
            )

        # Get package
        package = self._packages.get(listing.package_id)
        if not package:
            # Need to fetch from remote
            peer = self._peers.get(listing.publisher_node)
            if not peer:
                raise ValueError(f"Publisher node not found: {listing.publisher_node}")

            request = await self._transfer.request_transfer(listing_id, peer)
            # In production, would wait for transfer to complete
            raise ValueError("Remote transfer not yet implemented")

        # Update download count
        listing.download_count += 1

        return package.to_generator()

    def acquire_sync(self, listing_id: str) -> Any:
        """Synchronous acquire (local only).

        Args:
            listing_id: Listing to acquire

        Returns:
            Generator instance
        """
        listing = self._listings.get(listing_id)
        if not listing:
            raise ValueError(f"Listing not found: {listing_id}")

        package = self._packages.get(listing.package_id)
        if not package:
            raise ValueError(f"Package not found locally: {listing.package_id}")

        listing.download_count += 1
        return package.to_generator()

    def add_peer(self, peer: PeerNode) -> None:
        """Add a peer node to the network.

        Args:
            peer: Peer node configuration
        """
        self._peers[peer.node_id] = peer
        logger.info(f"Added peer: {peer.node_id} at {peer.address}")

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer node.

        Args:
            node_id: Peer node ID
        """
        if node_id in self._peers:
            del self._peers[node_id]

    def get_listing(self, listing_id: str) -> Optional[FederatedListing]:
        """Get a specific listing.

        Args:
            listing_id: Listing ID

        Returns:
            FederatedListing or None
        """
        return self._listings.get(listing_id)

    def get_my_listings(self) -> List[FederatedListing]:
        """Get listings published by this node.

        Returns:
            List of own listings
        """
        return [l for l in self._listings.values() if l.publisher_node == self.config.node_id]

    def get_balance(self) -> int:
        """Get current credit balance.

        Returns:
            Credit balance
        """
        return self._ledger.get_balance(self.config.node_id)

    def rate_listing(self, listing_id: str, rating: float, review: str = "") -> bool:
        """Rate a listing.

        Args:
            listing_id: Listing to rate
            rating: Rating (1-5)
            review: Optional review text

        Returns:
            True if rating was recorded
        """
        if listing_id not in self._listings:
            return False

        listing = self._listings[listing_id]

        # Update rating (simple average)
        old_total = listing.rating * listing.rating_count
        listing.rating_count += 1
        listing.rating = (old_total + rating) / listing.rating_count
        listing.updated_at = datetime.utcnow().isoformat()

        return True

    def verify_listing(self, listing_id: str) -> bool:
        """Verify a listing's package integrity.

        Args:
            listing_id: Listing to verify

        Returns:
            True if verification passed
        """
        listing = self._listings.get(listing_id)
        if not listing:
            return False

        package = self._packages.get(listing.package_id)
        if not package:
            return False

        # Verify signatures
        if self._transfer._verify_package(package):
            listing.verification_status = VerificationStatus.VERIFIED
            return True

        listing.verification_status = VerificationStatus.REJECTED
        return False

    def _save_state(self) -> None:
        """Save marketplace state to disk."""
        state_file = Path(self.config.storage_path) / "state.json"

        state = {
            "listings": {
                lid: {
                    "listing_id": l.listing_id,
                    "package_id": l.package_id,
                    "metadata": {
                        "name": l.metadata.name,
                        "description": l.metadata.description,
                        "domain": l.metadata.domain,
                    },
                    "publisher_node": l.publisher_node,
                    "price_credits": l.price_credits,
                    "license": l.license.value,
                    "download_count": l.download_count,
                    "rating": l.rating,
                }
                for lid, l in self._listings.items()
            },
            "balance": self._ledger.get_balance(self.config.node_id),
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load marketplace state from disk."""
        state_file = Path(self.config.storage_path) / "state.json"

        if not state_file.exists():
            return

        with open(state_file) as f:
            state = json.load(f)

        # Restore balance
        if "balance" in state:
            self._ledger.set_balance(self.config.node_id, state["balance"])

    def get_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "node_id": self.config.node_id,
            "total_listings": len(self._listings),
            "total_packages": len(self._packages),
            "connected_peers": len([p for p in self._peers.values() if p.is_online]),
            "credit_balance": self.get_balance(),
            "total_downloads": sum(l.download_count for l in self._listings.values()),
        }


# Convenience functions
def create_federated_marketplace(
    node_id: str,
    peers: Optional[List[str]] = None,
    storage_path: str = "./federated_marketplace",
) -> FederatedMarketplace:
    """Create a new federated marketplace node.

    Args:
        node_id: Unique node identifier
        peers: List of peer addresses
        storage_path: Local storage directory

    Returns:
        FederatedMarketplace instance
    """
    config = NodeConfig(
        node_id=node_id,
        peers=peers or [],
        storage_path=storage_path,
    )
    return FederatedMarketplace(config)

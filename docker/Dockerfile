# ESPectre Sensing Server
# Multi-stage build for minimal runtime image

# Stage 1: Build
FROM rust:1.85-bookworm AS builder
WORKDIR /build

COPY Cargo.toml Cargo.lock* ./
COPY server/ ./server/
COPY wifiscan/ ./wifiscan/

RUN cargo build --release -p espectre-server 2>&1     && strip target/release/espectre-server

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends     ca-certificates     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/espectre-server /app/espectre-server
COPY ui/ /app/ui/

# DL model files
COPY server/models/ /app/models/

# HTTP API
EXPOSE 4000
# WebSocket
EXPOSE 4001
# ESP32 UDP
EXPOSE 5005/udp

ENV RUST_LOG=info
ENV CSI_SOURCE=auto

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["/app/espectre-server --source ${CSI_SOURCE} --tick-ms 100 --ui-path /app/ui --http-port 4000 --ws-port 4001 --bind-addr 0.0.0.0"]

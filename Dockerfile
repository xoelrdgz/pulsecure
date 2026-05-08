FROM node:24-bookworm-slim AS web-builder
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/index.html web/tsconfig.json web/vite.config.ts ./
COPY web/src ./src
RUN npm run build

FROM rust:1.92-slim-bookworm AS rust-builder
WORKDIR /build
ARG CARGO_BUILD_JOBS=2
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY models ./models
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/usr/local/cargo/git --mount=type=cache,target=/build/target cargo build --release --bin pulsecure && mkdir -p /build/out && cp target/release/pulsecure /build/out/pulsecure

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y libsqlite3-0 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 pulsecure
WORKDIR /app
COPY models ./models/
COPY --from=rust-builder /build/out/pulsecure ./
COPY --from=web-builder /build/web/dist ./web/dist/
RUN mkdir -p /app/data && chown -R pulsecure:pulsecure /app
USER pulsecure
ENV RUST_LOG=info
ENV PULSECURE_WEB_ADDR=0.0.0.0:8080
ENV PULSECURE_DB_PATH=/app/data/pulsecure.db
EXPOSE 8080
ENTRYPOINT ["/app/pulsecure"]

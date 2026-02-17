# --- Stage 1: Build Environment (Ubuntu 24.04 with Go & Node.js) ---
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js, Go, and build tools
RUN apt-get update && apt-get install -y \
    golang \
    nodejs \
    npm \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Build the application
# Executes the clean and all targets in your Makefile
RUN CGO_ENABLED=0 make clean all

# --- Stage 2: Runtime Environment ---
FROM llama-server:latest

# Build arguments for permissions
ARG UID=1001
ARG GID=1001
ARG USER_HOME=/app

# Environment setup
ENV HOME=$USER_HOME
ENV PATH="/app:${PATH}"

# Switch to root to handle user creation and permissions
USER root

# Create the app group and user if they don't exist
RUN if [ $UID -ne 0 ]; then \
    groupadd --system --gid $GID app || true; \
    useradd --system --uid $UID --gid $GID --create-home --home-dir $USER_HOME app || true; \
    fi

# Ensure the /app directory exists and is owned by the app user
RUN mkdir -p /app && chown -R $UID:$GID /app

# Copy the specific build output from the builder stage
# Maps build/llama-swap-linux-amd64 to /app/llama-swap
COPY --from=builder --chown=$UID:$GID /app/build/llama-swap-linux-amd64 /app/llama-swap
COPY --from=builder --chown=$UID:$GID /app/config.example.yaml /app/config.yaml

# Set workdir and drop privileges
WORKDIR /app
USER $UID

# Healthcheck and Entrypoint
HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]


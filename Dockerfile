FROM python:3.12-bookworm

# Installer les dépendances système (OpenSlide, make, curl pour uv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    make \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt ./
RUN uv pip install --system -r requirements.txt

# Commande par défaut
CMD ["tail", "-f", "/dev/null"]

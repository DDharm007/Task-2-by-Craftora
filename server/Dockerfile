# ─── Build stage ─────────────────────────────────────────────────────────────
FROM node:22-slim AS build

WORKDIR /app

# Copy only manifests first so `npm ci` is cached until a dependency changes.
COPY package.json package-lock.json ./
COPY shared/package.json ./shared/
COPY server/package.json ./server/
COPY client/package.json ./client/

# The client is not built in this image; skip its (large) dependency tree.
RUN npm ci --workspace shared --workspace server --include-workspace-root

COPY shared ./shared
COPY server ./server

RUN npm run build --workspace shared \
 && npm run build --workspace server

# Drop dev dependencies from the tree we are about to copy forward.
RUN npm prune --omit=dev --workspace shared --workspace server --include-workspace-root

# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM node:22-slim AS runtime

ENV NODE_ENV=production \
    PORT=8787

WORKDIR /app

# `tini` reaps zombies and forwards SIGTERM, so graceful shutdown works.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Run unprivileged. The node image already ships a `node` user.
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/package.json ./package.json
COPY --from=build --chown=node:node /app/shared/dist ./shared/dist
COPY --from=build --chown=node:node /app/shared/package.json ./shared/package.json
COPY --from=build --chown=node:node /app/server/dist ./server/dist
COPY --from=build --chown=node:node /app/server/package.json ./server/package.json

# Writable mount points for the local vector store, model cache and dataset.
RUN mkdir -p /app/storage /app/models /app/dataset /app/uploads \
 && chown -R node:node /app/storage /app/models /app/dataset /app/uploads

# The prebuilt index, straight from the build context. Without it the embedded
# store loads zero vectors and every query retrieves nothing — the container
# filesystem is ephemeral, so there is nothing to index into at runtime.
COPY --chown=node:node storage/vectors /app/storage/vectors

USER node

EXPOSE 8787

# Uses the app's own health endpoint, so the check exercises the real stack.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||8787)+'/api/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["node", "server/dist/index.js"]

/**
 * Server entrypoint.
 *
 * Starts listening immediately and warms the models in the background, so the
 * health endpoint answers during a cold start instead of the process looking
 * hung while ONNX weights load.
 */
import { createApp } from './app.js';
import { config, redactedConfig } from './config/env.js';
import { logger } from './utils/logger.js';
import { getEmbeddingProvider } from './rag/embeddings/index.js';
import { getReranker } from './rag/reranker/index.js';
import { closeVectorStore, getVectorStore } from './rag/vector/index.js';
import { getIndexStats } from './services/indexing.service.js';

const app = createApp();

const server = app.listen(config.port, () => {
  logger.info(
    { port: config.port, env: config.env, cors: config.corsOrigins },
    `GoaRAG API listening on http://localhost:${config.port}`,
  );
  if (config.isDev) logger.debug({ config: redactedConfig() }, 'Effective configuration');
});

// SSE answers can run for minutes on a reasoning model; the default 2-minute
// header timeout would sever them mid-stream.
server.headersTimeout = 0;
server.requestTimeout = 0;
server.keepAliveTimeout = 76_000;

/** Warm the slow paths without blocking startup. */
async function warmup(): Promise<void> {
  try {
    const store = await getVectorStore();
    const stats = await getIndexStats();

    if (!stats.indexed) {
      logger.warn(
        { store: store.name },
        'Vector index is empty — run `npm run index` before querying',
      );
    } else {
      logger.info(
        {
          store: store.name,
          vectors: stats.vectors,
          documents: stats.documents,
          languages: stats.languages.slice(0, 5),
        },
        'Index ready',
      );
    }

    // Loading ONNX weights takes seconds; do it now so the first user query
    // does not pay for it.
    await getEmbeddingProvider().warmup();
    logger.info('Embedding model warm');

    await getReranker().warmup();
    logger.info('Reranker warm');
  } catch (error) {
    logger.error({ error: (error as Error).message }, 'Warmup failed — the API is up but degraded');
  }
}

void warmup();

/** Drain in-flight requests, then flush the vector store before exiting. */
async function shutdown(signal: string): Promise<void> {
  logger.info({ signal }, 'Shutting down');

  const forced = setTimeout(() => {
    logger.error('Graceful shutdown timed out — forcing exit');
    process.exit(1);
  }, 15_000);
  forced.unref();

  server.close(async () => {
    try {
      await closeVectorStore();
      logger.info('Shutdown complete');
      process.exit(0);
    } catch (error) {
      logger.error({ error: (error as Error).message }, 'Error during shutdown');
      process.exit(1);
    }
  });
}

process.on('SIGTERM', () => void shutdown('SIGTERM'));
process.on('SIGINT', () => void shutdown('SIGINT'));

process.on('unhandledRejection', (reason) => {
  logger.error({ reason: reason instanceof Error ? reason.message : String(reason) }, 'Unhandled rejection');
});

process.on('uncaughtException', (error) => {
  logger.fatal({ error: error.message, stack: error.stack }, 'Uncaught exception — exiting');
  process.exit(1);
});

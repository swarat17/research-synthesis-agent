-- Run once in the Supabase SQL editor for your project
CREATE TABLE IF NOT EXISTS query_logs (
    query_id          TEXT PRIMARY KEY,
    timestamp         TIMESTAMPTZ DEFAULT NOW(),
    query             TEXT,
    total_cost_usd    FLOAT8,
    total_latency_ms  FLOAT8,
    num_papers        INT,
    num_contradictions INT,
    num_hypotheses    INT,
    node_breakdown    JSONB
);

-- Optional: index for fast recent-query lookups
CREATE INDEX IF NOT EXISTS query_logs_timestamp_idx ON query_logs (timestamp DESC);

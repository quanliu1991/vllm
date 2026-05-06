#!/bin/bash
set -e

# Default: start both services
START_LLM="${START_LLM:-true}"
START_EMBEDDING="${START_EMBEDDING:-true}"

SUPERVISORD_CONF_BASE="/etc/supervisor/supervisord.conf"
SUPERVISORD_CONF_TMP="/tmp/supervisord.conf"

echo "[INFO] Starting with START_LLM=$START_LLM, START_EMBEDDING=$START_EMBEDDING"

# Build the supervisord config dynamically
cat > "$SUPERVISORD_CONF_TMP" <<'EOF'
[supervisord]
nodaemon=true
logfile=/workspace/logs/supervisord.log
logfile_maxbytes=50MB
logfile_backups=3
loglevel=info
pidfile=/tmp/supervisord.pid
user=root

[unix_http_server]
file=/tmp/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock

EOF

if [ "$START_LLM" = "true" ]; then
    cat >> "$SUPERVISORD_CONF_TMP" <<'EOF'
[program:llm]
command=/workspace/hb-serve/run.sh
autorestart=true
startretries=3
startsecs=10
stdout_logfile=/workspace/logs/llm.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
stderr_logfile=/workspace/logs/llm.err
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=3
priority=10

EOF
    echo "[INFO] LLM service enabled"
else
    echo "[INFO] LLM service disabled"
fi

if [ "$START_EMBEDDING" = "true" ]; then
    cat >> "$SUPERVISORD_CONF_TMP" <<'EOF'
[program:embedding]
command=/workspace/services/embedding/run_embed.sh
autorestart=true
startretries=3
startsecs=10
stdout_logfile=/workspace/logs/embedding.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
stderr_logfile=/workspace/logs/embedding.err
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=3
priority=20

EOF
    echo "[INFO] Embedding service enabled"
else
    echo "[INFO] Embedding service disabled"
fi

# Verify at least one service is enabled
if [ "$START_LLM" != "true" ] && [ "$START_EMBEDDING" != "true" ]; then
    echo "[ERROR] Both START_LLM and START_EMBEDDING are disabled. At least one must be true."
    exit 1
fi

echo "[INFO] Generated supervisord config at $SUPERVISORD_CONF_TMP"
cat "$SUPERVISORD_CONF_TMP"
echo "---"

# Start supervisord with the generated config
exec supervisord -c "$SUPERVISORD_CONF_TMP"

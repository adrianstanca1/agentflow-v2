FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Production image ──────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf 2>/dev/null || true

# Runtime env injection
RUN echo '#!/bin/sh\n\
CONFIG_FILE="/usr/share/nginx/html/env-config.js"\n\
echo "window.__ENV = {" > $CONFIG_FILE\n\
echo "  API_URL: \"${API_URL:-}\"," >> $CONFIG_FILE\n\
echo "  WS_URL: \"${WS_URL:-}\"" >> $CONFIG_FILE\n\
echo "};" >> $CONFIG_FILE\n\
exec nginx -g "daemon off;"' > /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

EXPOSE 80
CMD ["/docker-entrypoint.sh"]

The proxy runs from the prebuilt image `public.ecr.aws/aktosecurity/akto-egress-proxy:latest`, which bundles `akto_guardrails.py`. Run the following - 

```
export ANTHROPIC_API_KEY=sk-ant-....
export AKTO_URL=...
export APP_NAME=...        # identifies your app in Akto guardrails (sent as the host header)
export AKTO_TOKEN=...      # needed when data-ingestion runs with authentication on
export AKTO_TAGS=env=prod,team=payments   # optional extra tags sent to Akto
docker compose up --build
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for the agent |
| `AKTO_URL` | Yes | Base URL of your Akto instance (e.g. `https://akto.example.com`). |
| `APP_NAME` | Yes | Name of your application. When set, it is sent as the `host` header in requests to Akto so you can identify traffic per app. |
| `AKTO_TOKEN` | When data-ingestion authentication is on | Data-ingestion token (the account's database-abstractor JWT). Sent as a bare JWT in the `Authorization` header, with no `Bearer ` prefix. |
| `AI_HOSTS` | No | Comma-separated hosts to intercept and run guardrails on; `*` matches any run of characters. Defaults to `api.openai.com,api.anthropic.com,chatgpt.com,bedrock-runtime.*.amazonaws.com,bedrock-agent-runtime.*.amazonaws.com`. All other traffic is tunnelled without inspection. |
| `AKTO_TAGS` | No | Comma-separated `key=value` pairs added to the `tag` sent to Akto (e.g. `env=prod,team=payments`). The built-in `gen-ai` and `source` tags are always sent and cannot be overridden. |
| `AKTO_FILE_GUARDRAILS` | No | On by default; `false` turns it off. Checks the files in AI requests: multipart uploads (e.g. `/v1/files`) and base64 images and documents in JSON bodies (OpenAI, Anthropic, Gemini, Bedrock Converse). Each request's files are sent to `${AKTO_URL}/api/validate/file`, and a blocked file blocks the request with a 403. The text guardrails see a `[akto-file sha256:…]` placeholder in place of each base64 file. Requests without a file, or whose files were allowed recently (see `AKTO_FILE_CACHE_TTL`), make no file call. Needs file validation on in guardrails-service (`FILE_VALIDATION_ENABLED`, on by default). |
| `AKTO_FILE_MAX_BYTES` | No | Files larger than this are not checked. Default `20971520` (20 MB). |
| `AKTO_FILE_TIMEOUT` | No | Seconds to wait for a file verdict. Default `30`. Keep it below data-ingestion's `GUARDRAILS_FILE_CLIENT_TIMEOUT_MS`. |
| `AKTO_FILE_FAIL_OPEN` | No | `false` blocks the request when the file check fails or times out. Default `true` (allow), like the text guardrails. |
| `AKTO_FILE_CACHE_TTL` | No | Seconds an allowed file (by sha256) is not checked again, since agents re-send their history every turn. A policy change reaches a remembered file only after this. `0` turns the cache off. Default `3600`. Blocked files are never remembered. |
| `AKTO_FILE_CACHE_SIZE` | No | Most allowed files remembered at once; the least recently used is dropped first. Default `1024`. |

To add a provider, add it to `AI_HOSTS` and to the `--ignore-hosts` regex in `docker-compose.yml`. A host is intercepted only if it is in both.

## Publishing the proxy image

After changing `akto-egress-proxy/`, rebuild and push the image. ECR Public is managed from `us-east-1`:

```
aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin public.ecr.aws
docker buildx build --platform linux/amd64,linux/arm64 \
  -t public.ecr.aws/aktosecurity/akto-egress-proxy:latest \
  --push ./akto-egress-proxy
```

# J.A.R.V.I.S. Security & Agent Foundation

## Security model implemented in phase 1

1. The browser requests a new credential from `POST /api/session`.
2. The backend generates a high-entropy session ID and bearer token.
3. Only a SHA-256 digest of the bearer token is retained in backend memory.
4. Every protected API request must send `Authorization: Bearer <token>`.
5. The backend derives identity from the token and rejects a different claimed
   `session_id` with HTTP 403.
6. Credentials expire after `SESSION_TOKEN_TTL_SECONDS` of inactivity and are
   invalidated by a backend restart.
7. Admin privilege is memory-only, expires independently, and is never restored
   from session metadata on disk.

## Endpoint policy

Public endpoints are limited to:

- `GET /api/health` — sanitized liveness information
- `POST /api/session` — rate-limited session bootstrap
- `GET /api/security/status` — sanitized lockdown state
- `POST /api/security/clear` — rate-limited break-glass flow

All other `/api/*` endpoints require a valid bearer token. Sensitive global
information and actions (system logs, token usage, knowledge internals, training,
OS access, model switching, and security controls) additionally require admin.

## Agent action policy

- PC command execution and raw shell execution default to disabled.
- The default OPEN allowlist contains only `notepad` and `calc`.
- If raw shell is deliberately enabled, every command is held in memory as a
  proposed action with an action ID.
- Execution requires a fresh admin-password confirmation and expires after two
  minutes. Destructive whole-drive commands remain blocked separately.

## Browser policy

- Credentials use `sessionStorage`, not persistent `localStorage`.
- The frontend automatically bootstraps a server-issued session.
- A 401 response creates a fresh session and retries only after the backend has
  rejected the original request, so the action was not processed twice.
- CORS is restricted to configured frontend and local development origins.

## Deployment requirement

The frontend and backend changes must be deployed together. Restarting only the
backend will make an older frontend receive HTTP 401 because it does not send a
bearer token.

Before an enterprise rollout, replace the Quick Tunnel with a named Cloudflare
Tunnel protected by Cloudflare Access or organizational SSO. Phase 1 bearer
sessions isolate users but are not a substitute for employee identity, RBAC,
offboarding, or central audit.

## Next foundation milestones

- Organizational SSO and named user identities
- RBAC and document-level access control
- PostgreSQL/Redis-backed sessions, rate limits, and immutable audit events
- Structured tools instead of regex command tags
- Approval UI showing action, target, risk, and expected data changes
- Sandboxed worker for tool execution
- RAG with document ACLs and citations
- Prompt-injection, upload-fuzzing, concurrency, and end-to-end browser tests

## Phase 2 foundation now included

- Optional Cloudflare Access identity mapping and organizational domain checks
- RBAC roles: staff, manager, and admin
- Role-aware local RAG with `[KB:filename#chunk-N]` citations
- Append-only SQLite audit events with SHA-256 hash-chain verification
- PostgreSQL immutable audit migration
- Structured tool schemas with role and argument validation
- Restricted read-only tool worker with no generic command operation
- Named Tunnel configuration template and launch script

Named Tunnel and SSO enforcement remain disabled until a Cloudflare account
administrator completes `cloudflared tunnel login`, provisions a hostname, and
creates the Access application and policies.

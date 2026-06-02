/**
 * TeNNet-SAC MCP Server — Cloudflare Workers
 *
 * MCP Streamable HTTP transport (protocol version 2025-03-26).
 * Stateless — no Durable Objects or session management required.
 *
 * POST /   — MCP JSON-RPC requests.
 *            Responds as SSE (text/event-stream) when client sends
 *            Accept: text/event-stream, otherwise plain JSON.
 * GET  /   — Opens an SSE stream for server-initiated messages.
 *            Returns an empty stream immediately (this server is stateless
 *            and never pushes server-initiated messages).
 * DELETE / — Close session (no-op for stateless server, always succeeds).
 *
 * ⚠️  Timeout note
 *   Regular tools           : up to 180 s — fine on Workers Paid (Unbound).
 *   compute_nrtl_parameters : up to 600 s — may exceed Workers wall-clock
 *   limits on some plans.  Route NRTL calls through a queue worker if needed.
 *
 * Deploy
 *   npm install
 *   npx wrangler deploy
 *
 * Local dev
 *   npx wrangler dev
 */

export interface Env {
  TENNETSAC_API?: string;
}

const PROTOCOL_VERSION = "2025-03-26";
const DEFAULT_API = "https://stlin-tennetsac.hf.space";
const TIMEOUT_MS = 180_000;
const NRTL_TIMEOUT_MS = 600_000;

// ---------------------------------------------------------------------------
// MCP Tool definitions (JSON Schema format required by the MCP spec)
// ---------------------------------------------------------------------------

const TOOLS = [
  {
    name: "compute_sigma_profile",
    description:
      "Compute the sigma (σ) profile, surface area, and molecular volume " +
      "for a molecule from its SMILES string.\n\n" +
      "Returns: sigma (e/Å²), p_sigma (Å²), area (Å²), volume (Å³).",
    inputSchema: {
      type: "object",
      properties: {
        smiles: {
          type: "string",
          description: "SMILES string of the molecule (e.g. 'CCO' for ethanol)",
        },
      },
      required: ["smiles"],
    },
  },
  {
    name: "compute_binary_activity_coefficients",
    description:
      "Compute ln(γ) activity coefficients and ensemble uncertainties for a " +
      "binary mixture across a list of mole fractions of component 1.\n\n" +
      "Returns: x1, lng1, lng2, lng1_std, lng2_std.",
    inputSchema: {
      type: "object",
      properties: {
        smiles1: { type: "string", description: "SMILES string for component 1" },
        smiles2: { type: "string", description: "SMILES string for component 2" },
        temperature: {
          type: "number",
          description: "Temperature in Kelvin (must be > 0)",
        },
        x1_values: {
          type: "array",
          items: { type: "number" },
          description: "List of x₁ mole fractions (0–1), e.g. [0.0, 0.25, 0.5, 0.75, 1.0]",
        },
      },
      required: ["smiles1", "smiles2", "temperature", "x1_values"],
    },
  },
  {
    name: "compute_binary_single_point",
    description:
      "Compute ln(γ) for a binary mixture at a single composition, including " +
      "sigma profiles and segment activity coefficients (ln Γ) for each component.\n\n" +
      "Returns: lng1, lng2, lng1_std, lng2_std, sigma_grid, sigma_profile, segac.",
    inputSchema: {
      type: "object",
      properties: {
        smiles1: { type: "string", description: "SMILES string for component 1" },
        smiles2: { type: "string", description: "SMILES string for component 2" },
        temperature: { type: "number", description: "Temperature in Kelvin" },
        x1: {
          type: "number",
          description: "Mole fraction of component 1 (0 to 1)",
        },
      },
      required: ["smiles1", "smiles2", "temperature", "x1"],
    },
  },
  {
    name: "compute_multicomponent_activity_coefficients",
    description:
      "Compute ln(γ) activity coefficients for a multicomponent mixture " +
      "(minimum 3 components).\n\n" +
      "Returns: smiles, x, lng, lng_std, sigma_grid, sigma_profile, segac.",
    inputSchema: {
      type: "object",
      properties: {
        smiles: {
          type: "array",
          items: { type: "string" },
          description: "List of SMILES strings — must have at least 3 entries",
        },
        temperature: { type: "number", description: "Temperature in Kelvin" },
        x_values: {
          type: "array",
          items: { type: "number" },
          description:
            "Mole fractions for the first N-1 components; the last is inferred " +
            "(values must sum to < 1)",
        },
      },
      required: ["smiles", "temperature", "x_values"],
    },
  },
  {
    name: "compute_nrtl_parameters",
    description:
      "Fit Aspen Plus NRTL binary interaction parameters (aij, bij, cij, dij, eij, fij) " +
      "from TeNNet-SAC predictions.\n\n" +
      "Full model: τij(T) = aij + bij/T + eij·ln(T) + fij·T;  αij(T) = cij + dij·(T−273.15).\n\n" +
      "⚠️ Computation time scales with NT × n_points; defaults (NT=6, n_points=21) " +
      "take roughly 1–3 minutes on the HuggingFace free tier.",
    inputSchema: {
      type: "object",
      properties: {
        smiles1:  { type: "string",  description: "SMILES string for component i" },
        smiles2:  { type: "string",  description: "SMILES string for component j" },
        Ti:       { type: "number",  description: "Lower temperature bound (K), default 250" },
        Tf:       { type: "number",  description: "Upper temperature bound (K), default 600" },
        NT:       { type: "integer", description: "Number of fitting temperatures (3–20), default 6" },
        n_points: { type: "integer", description: "Composition grid points per temperature (5–51), default 21" },
        c:        { type: "number",  description: "Initial/fixed cij (non-randomness base), default 0.3" },
        d:        { type: "number",  description: "Initial/fixed dij (T-coefficient of α, K⁻¹), default 0" },
        fit_c:    { type: "boolean", description: "If true, optimise cij (triggers global nonlinear fit)" },
        fit_d:    { type: "boolean", description: "If true, optimise dij (implies fit_c; α becomes T-dependent)" },
        fit_e:    { type: "boolean", description: "If true, include eij·ln(T) term in τ(T)" },
        fit_f:    { type: "boolean", description: "If true, include fij·T term in τ(T)" },
      },
      required: ["smiles1", "smiles2"],
    },
  },
] as const;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Mcp-Session-Id, Accept",
  "Access-Control-Expose-Headers": "Mcp-Session-Id",
};

/** True when the client accepts an SSE stream in response. */
function wantsSSE(request: Request): boolean {
  return (request.headers.get("Accept") ?? "").includes("text/event-stream");
}

/** Wrap a JSON-RPC object as a single SSE event and close the stream. */
function sseResp(body: unknown, status = 200): Response {
  const payload = `event: message\ndata: ${JSON.stringify(body)}\n\n`;
  return new Response(payload, {
    status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      ...CORS_HEADERS,
    },
  });
}

/** Plain JSON response. */
function jsonResp(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

function rpcResult(id: unknown, result: unknown, sse: boolean): Response {
  const msg = { jsonrpc: "2.0", id, result };
  return sse ? sseResp(msg) : jsonResp(msg);
}

function rpcError(id: unknown, code: number, message: string, sse = false): Response {
  const msg = { jsonrpc: "2.0", id, error: { code, message } };
  return sse ? sseResp(msg) : jsonResp(msg);
}

// ---------------------------------------------------------------------------
// Upstream API call
// ---------------------------------------------------------------------------

async function apiPost(
  base: string,
  path: string,
  body: unknown,
  timeoutMs: number,
): Promise<unknown> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Upstream error ${res.status}: ${text}`);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

async function callTool(
  name: string,
  args: Record<string, unknown>,
  apiBase: string,
): Promise<unknown> {
  switch (name) {
    case "compute_sigma_profile":
      return apiPost(apiBase, "/api/profile", { smiles: args.smiles }, TIMEOUT_MS);

    case "compute_binary_activity_coefficients":
      return apiPost(
        apiBase,
        "/api/binary",
        {
          smiles: [args.smiles1, args.smiles2],
          temperature: args.temperature,
          x1_values: args.x1_values,
        },
        TIMEOUT_MS,
      );

    case "compute_binary_single_point":
      return apiPost(
        apiBase,
        "/api/binary_single",
        {
          smiles: [args.smiles1, args.smiles2],
          temperature: args.temperature,
          x1: args.x1,
        },
        TIMEOUT_MS,
      );

    case "compute_multicomponent_activity_coefficients":
      return apiPost(
        apiBase,
        "/api/multicomponent",
        {
          smiles: args.smiles,
          temperature: args.temperature,
          x_values: args.x_values,
        },
        TIMEOUT_MS,
      );

    case "compute_nrtl_parameters": {
      const body: Record<string, unknown> = {
        smiles: [args.smiles1, args.smiles2],
      };
      for (const key of ["Ti", "Tf", "NT", "n_points", "c", "d", "fit_c", "fit_d", "fit_e", "fit_f"]) {
        if (key in args) body[key] = args[key];
      }
      const data = (await apiPost(apiBase, "/api/nrtl_fit", body, NRTL_TIMEOUT_MS)) as Record<
        string,
        unknown
      >;
      delete data["verification"];
      return data;
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ---------------------------------------------------------------------------
// Worker entry point
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const apiBase = (env.TENNETSAC_API ?? DEFAULT_API).replace(/\/$/, "");

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // GET: SSE channel for server-initiated messages.
    // This server is stateless and never sends server-initiated messages,
    // so we return an empty stream that closes immediately.
    if (request.method === "GET") {
      return new Response("", {
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache, no-transform",
          ...CORS_HEADERS,
        },
      });
    }

    // DELETE: close session (no-op for a stateless server).
    if (request.method === "DELETE") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (request.method !== "POST") {
      return new Response("Method Not Allowed", {
        status: 405,
        headers: CORS_HEADERS,
      });
    }

    const sse = wantsSSE(request);

    // Parse JSON-RPC envelope
    type RpcMsg = { jsonrpc?: string; id?: unknown; method?: string; params?: unknown };
    let msg: RpcMsg;
    try {
      msg = (await request.json()) as RpcMsg;
    } catch {
      return rpcError(null, -32700, "Parse error", sse);
    }

    if (msg.jsonrpc !== "2.0" || typeof msg.method !== "string") {
      return rpcError(msg.id ?? null, -32600, "Invalid Request", sse);
    }

    const { id, method, params } = msg;

    switch (method) {
      // ── Lifecycle ────────────────────────────────────────────────────────
      case "initialize":
        return rpcResult(id, {
          protocolVersion: PROTOCOL_VERSION,
          capabilities: { tools: {} },
          serverInfo: { name: "TeNNet-SAC", version: "1.0.0" },
        }, sse);

      case "notifications/initialized":
        return new Response(null, { status: 204, headers: CORS_HEADERS });

      case "ping":
        return rpcResult(id, {}, sse);

      // ── Tools ─────────────────────────────────────────────────────────────
      case "tools/list":
        return rpcResult(id, { tools: TOOLS }, sse);

      case "tools/call": {
        const { name, arguments: toolArgs } = params as {
          name: string;
          arguments: Record<string, unknown>;
        };
        try {
          const data = await callTool(name, toolArgs ?? {}, apiBase);
          return rpcResult(id, {
            content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
          }, sse);
        } catch (err) {
          return rpcResult(id, {
            content: [{ type: "text", text: String(err) }],
            isError: true,
          }, sse);
        }
      }

      default:
        return rpcError(id, -32601, "Method not found", sse);
    }
  },
};

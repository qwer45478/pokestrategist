const DEFAULT_CONFIG = {
  enabled: true,
  endpoint: "http://127.0.0.1:8765/api/suggest",
  refreshMs: 1500,
  requestTimeoutMs: 5000
};

async function getConfig() {
  const stored = await chrome.storage.sync.get(DEFAULT_CONFIG);
  return { ...DEFAULT_CONFIG, ...stored };
}

async function saveConfig(configPatch) {
  const nextConfig = { ...(await getConfig()), ...configPatch };
  await chrome.storage.sync.set(nextConfig);
  return nextConfig;
}

function timeoutSignal(timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return { signal: controller.signal, clear: () => clearTimeout(timer) };
}

async function fetchSuggestion(snapshot) {
  const config = await getConfig();
  if (!config.enabled) {
    return {
      ok: false,
      source: "disabled",
      error: "pokestrategist companion is disabled in popup settings."
    };
  }

  const { signal, clear } = timeoutSignal(config.requestTimeoutMs);
  try {
    const response = await fetch(config.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ snapshot }),
      signal
    });

    if (!response.ok) {
      return {
        ok: false,
        source: "remote",
        error: `Local endpoint returned ${response.status}.`
      };
    }

    const payload = await response.json();
    return {
      ok: true,
      source: payload.source || "remote",
      suggestions: Array.isArray(payload.suggestions) ? payload.suggestions : [],
      metadata: payload.metadata || null
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      source: "remote",
      error: message
    };
  } finally {
    clear();
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const handler = async () => {
    if (message?.type === "pokestrategist:get-config") {
      return getConfig();
    }
    if (message?.type === "pokestrategist:save-config") {
      return saveConfig(message.payload || {});
    }
    if (message?.type === "pokestrategist:fetch-suggestion") {
      return fetchSuggestion(message.payload?.snapshot || null);
    }
    return null;
  };

  handler()
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      sendResponse({ ok: false, error: message });
    });

  return true;
});
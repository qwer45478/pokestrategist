async function sendMessage(type, payload) {
  const response = await chrome.runtime.sendMessage({ type, payload });
  if (!response?.ok) {
    throw new Error(response?.error || "Unknown extension error");
  }
  return response.result;
}

function setStatus(message) {
  const node = document.getElementById("statusText");
  if (node) {
    node.textContent = message;
  }
}

async function loadConfig() {
  const config = await sendMessage("pokestrategist:get-config");
  document.getElementById("enabled").checked = Boolean(config.enabled);
  document.getElementById("endpoint").value = config.endpoint || "";
  document.getElementById("refreshMs").value = String(config.refreshMs || 1500);
  setStatus("配置已加载。保存后刷新 Showdown 页面即可生效。");
}

async function saveConfig() {
  const payload = {
    enabled: document.getElementById("enabled").checked,
    endpoint: document.getElementById("endpoint").value.trim(),
    refreshMs: Number(document.getElementById("refreshMs").value || 1500)
  };
  await sendMessage("pokestrategist:save-config", payload);
  setStatus("配置已保存。若 Showdown 页面已打开，请刷新页面。 ");
}

document.getElementById("saveButton").addEventListener("click", () => {
  saveConfig().catch((error) => {
    setStatus(error instanceof Error ? error.message : String(error));
  });
});

loadConfig().catch((error) => {
  setStatus(error instanceof Error ? error.message : String(error));
});
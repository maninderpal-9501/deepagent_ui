export interface StandaloneConfig {
  deploymentUrl: string;
  assistantId: string;
  langsmithApiKey?: string;
}

const CONFIG_KEY = "deep-agent-config";

// Returns the config baked in via environment variables.
// Falls back to the local dev server defaults if the vars are not set.
export function getDefaultConfig(): StandaloneConfig {
  return {
    deploymentUrl:
      process.env.NEXT_PUBLIC_DEPLOYMENT_URL || "http://127.0.0.1:2024",
    assistantId: process.env.NEXT_PUBLIC_ASSISTANT_ID || "deep_agent",
  };
}

export function getConfig(): StandaloneConfig | null {
  if (typeof window === "undefined") return null;

  const stored = localStorage.getItem(CONFIG_KEY);
  if (!stored) return null;

  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

export function saveConfig(config: StandaloneConfig): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
}

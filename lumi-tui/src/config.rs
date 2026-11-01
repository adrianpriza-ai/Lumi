//! Resolve Lumi's configuration for the TUI: the same `.env`, `config.toml`
//! and `LUMI__LLM__*` overrides the Python side reads, so `lumi-tui` and
//! `lumi chat` always talk to the same model with the same keys.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4.1-mini";

#[derive(Debug, Clone)]
pub struct Config {
    /// OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
    /// Model id sent in `chat/completions`.
    pub model: String,
    /// API key for the LLM endpoint, if the env var holds one.
    pub api_key: Option<String>,
    /// Tavily keys (first is used; the pool stays a Python-side nicety).
    pub tavily_keys: Vec<String>,
    /// Firecrawl keys, same shape.
    pub firecrawl_keys: Vec<String>,
    /// Tool-call round trips per message, mirroring `[llm] max_tool_iterations`.
    pub max_tool_iterations: u32,
    /// Print the model's reasoning stream as dim text while it thinks.
    pub show_reasoning: bool,
    /// Where `.env` and `config.toml` were found.
    pub root: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
struct TomlLlm {
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    api_key_env: String,
    #[serde(default)]
    max_tool_iterations: u32,
}

#[derive(Debug, Default, Deserialize)]
struct TomlFile {
    #[serde(default)]
    llm: TomlLlm,
}

fn split_keys(raw: &str) -> Vec<String> {
    raw.split([',', ';'])
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .collect()
}

fn find_root() -> PathBuf {
    // Walk up from the current directory looking for config.toml, exactly
    // like the Python side resolves its paths from the pyproject directory.
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..6 {
        if dir.join("config.toml").is_file() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    PathBuf::from(".")
}

impl Config {
    pub fn load() -> Result<Self> {
        let root = find_root();
        let env_file = root.join(".env");
        if env_file.is_file() {
            dotenvy::from_path(&env_file)
                .with_context(|| format!("parsing {}", env_file.display()))?;
        }

        let toml_path = root.join("config.toml");
        let file: TomlFile = if toml_path.is_file() {
            let raw = std::fs::read_to_string(&toml_path)
                .with_context(|| format!("reading {}", toml_path.display()))?;
            toml::from_str(&raw).with_context(|| format!("parsing {}", toml_path.display()))?
        } else {
            TomlFile::default()
        };

        // LUMI__LLM__* beats config.toml, matching the Python override rule.
        let env_override = |key: &str| std::env::var(format!("LUMI__LLM__{}", key)).ok();

        let base_url = env_override("BASE_URL")
            .filter(|v| !v.trim().is_empty())
            .or_else(|| Some(file.llm.base_url.clone()).filter(|v| !v.trim().is_empty()))
            .or_else(|| std::env::var("OPENAI_BASE_URL").ok().filter(|v| !v.trim().is_empty()))
            .or_else(|| std::env::var("OPENAI_API_BASE").ok().filter(|v| !v.trim().is_empty()))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

        let model = env_override("MODEL")
            .filter(|v| !v.trim().is_empty())
            .or_else(|| Some(file.llm.model.clone()).filter(|v| !v.trim().is_empty()))
            .or_else(|| std::env::var("OPENAI_MODEL").ok().filter(|v| !v.trim().is_empty()))
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let key_env = env_override("API_KEY_ENV")
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| {
                if file.llm.api_key_env.trim().is_empty() {
                    "OPENAI_API_KEY".to_string()
                } else {
                    file.llm.api_key_env.trim().to_string()
                }
            });
        let api_key = std::env::var(&key_env)
            .ok()
            .and_then(|raw| split_keys(&raw).into_iter().next())
            .filter(|k| !k.is_empty() && !k.starts_with("sk-..."));

        let tavily_keys = std::env::var("TAVILY_API_KEY")
            .map(|raw| split_keys(&raw))
            .unwrap_or_default();
        let firecrawl_keys = std::env::var("FIRECRAWL_API_KEY")
            .map(|raw| split_keys(&raw))
            .unwrap_or_default();

        let show_reasoning = std::env::var("LUMI__LLM__SHOW_REASONING")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        let max_tool_iterations = env_override("MAX_TOOL_ITERATIONS")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| {
                let from_file = file.llm.max_tool_iterations;
                if from_file == 0 { 20 } else { from_file.clamp(1, 40) }
            });

        Ok(Self {
            base_url,
            model,
            api_key,
            tavily_keys,
            firecrawl_keys,
            max_tool_iterations,
            show_reasoning,
            root,
        })
    }

    /// Masked endpoint for the status bar: never shows the key, shows the host.
    pub fn host(&self) -> String {
        self.base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string()
    }

    pub fn search_status(&self) -> String {
        if !self.tavily_keys.is_empty() {
            "tavily".into()
        } else if !self.firecrawl_keys.is_empty() {
            "firecrawl".into()
        } else {
            "no search key".into()
        }
    }
}

/// Read a workspace file for the `read` tool / context, refusing to escape
/// the project root — the same promise the Python bot makes.
pub fn safe_join(root: &Path, relative: &str) -> Result<PathBuf> {
    let cleaned = relative.trim_start_matches(['/', '~']);
    let path = root.join(cleaned);
    let canonical = path
        .canonicalize()
        .map_err(|e| anyhow!("{}: {}", relative, e))?;
    let root_canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if !canonical.starts_with(&root_canonical) {
        return Err(anyhow!("{}: outside the project root", relative));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_key_pools() {
        assert_eq!(split_keys("a, b ;c"), vec!["a", "b", "c"]);
        assert!(split_keys("").is_empty());
    }

    #[test]
    fn safe_join_blocks_escape() {
        let root = Path::new("/tmp");
        assert!(safe_join(root, "../../etc/passwd").is_err());
    }
}

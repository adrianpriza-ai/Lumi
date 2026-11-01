//! Web access through Tavily and Firecrawl — the same two providers Lumi's
//! Python web tool prefers. Plain REST, no SDKs: Tavily returns cleaned,
//! LLM-ready snippets; Firecrawl adds page scraping for the `fetch_page` tool.

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::{json, Value};

fn tavily_url() -> String {
    std::env::var("TAVILY_API_URL").unwrap_or_else(|_| "https://api.tavily.com/search".into())
}

fn firecrawl_url(path: &str) -> String {
    let base =
        std::env::var("FIRECRAWL_API_URL").unwrap_or_else(|_| "https://api.firecrawl.dev".into());
    format!("{base}{path}")
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub struct Search {
    http: Client,
    tavily_keys: Vec<String>,
    firecrawl_keys: Vec<String>,
}

impl Search {
    pub fn new(tavily_keys: Vec<String>, firecrawl_keys: Vec<String>) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(15))
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("reqwest client"),
            tavily_keys,
            firecrawl_keys,
        }
    }

    /// Search, preferring Tavily, falling back to Firecrawl.
    pub async fn web_search(&self, query: &str, max_results: usize) -> Result<Vec<Hit>> {
        let max = max_results.clamp(1, 10);
        if !self.tavily_keys.is_empty() {
            match self.tavily_search(&self.tavily_keys[0], query, max).await {
                Ok(hits) => return Ok(hits),
                Err(e) if self.firecrawl_keys.is_empty() => return Err(e),
                Err(_) => {} // fall through to Firecrawl
            }
        }
        let key = self
            .firecrawl_keys
            .first()
            .ok_or_else(|| anyhow!("no web search key — set TAVILY_API_KEY in .env"))?;
        self.firecrawl_search(key, query, max).await
    }

    async fn tavily_search(&self, key: &str, query: &str, max: usize) -> Result<Vec<Hit>> {
        let response = self
            .http
            .post(tavily_url())
            .bearer_auth(key)
            .json(&json!({
                "query": query,
                "max_results": max,
                "include_answer": false,
                "search_depth": "basic",
            }))
            .send()
            .await?;
        let body = read_json(response, "tavily").await?;
        let mut hits = Vec::new();
        if let Some(results) = body.get("results").and_then(Value::as_array) {
            for result in results {
                hits.push(Hit {
                    title: str_of(result, "title"),
                    url: str_of(result, "url"),
                    snippet: str_of(result, "content"),
                });
            }
        }
        Ok(hits)
    }

    async fn firecrawl_search(&self, key: &str, query: &str, max: usize) -> Result<Vec<Hit>> {
        let response = self
            .http
            .post(firecrawl_url("/v1/search"))
            .bearer_auth(key)
            .json(&json!({ "query": query, "limit": max }))
            .send()
            .await?;
        let body = read_json(response, "firecrawl").await?;
        let mut hits = Vec::new();
        if let Some(results) = body
            .get("data")
            .and_then(|d| d.get("web"))
            .and_then(Value::as_array)
        {
            for result in results {
                hits.push(Hit {
                    title: str_of(result, "title"),
                    url: str_of(result, "url"),
                    snippet: str_of(result, "description"),
                });
            }
        }
        Ok(hits)
    }

    /// Scrape one page to markdown through Firecrawl.
    pub async fn fetch_page(&self, url: &str) -> Result<String> {
        let key = self
            .firecrawl_keys
            .first()
            .ok_or_else(|| anyhow!("fetch_page needs FIRECRAWL_API_KEY in .env"))?;
        let response = self
            .http
            .post(firecrawl_url("/v1/scrape"))
            .bearer_auth(key)
            .json(&json!({ "url": url, "formats": ["markdown"] }))
            .send()
            .await?;
        let body = read_json(response, "firecrawl").await?;
        let markdown = body
            .get("data")
            .and_then(|d| d.get("markdown"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if markdown.is_empty() {
            return Err(anyhow!("firecrawl returned no content for {}", url));
        }
        Ok(markdown)
    }
}

fn str_of(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn detail_message(body: &Value) -> String {
    body.get("error")
        .or_else(|| body.get("detail"))
        .or_else(|| body.get("message"))
        .map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))
        .unwrap_or_else(|| "unknown error".into())
}

/// Read a provider reply as JSON, failing with the real status and a snippet
/// of the body when it is not JSON (gateway HTML, empty 502s, …) instead of a
/// generic deserialization error.
async fn read_json(response: reqwest::Response, provider: &str) -> Result<Value> {
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    let body = serde_json::from_str(&text).map_err(|_| {
        anyhow!(
            "{} {} — non-JSON response: {}",
            provider,
            status,
            crate::llm::truncate(text.trim(), 200)
        )
    })?;
    if !status.is_success() {
        return Err(anyhow!("{} {} — {}", provider, status, detail_message(&body)));
    }
    Ok(body)
}

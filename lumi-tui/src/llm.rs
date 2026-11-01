//! OpenAI-compatible chat completions with SSE streaming and tool calls.
//! Works against OpenAI, Ollama, LM Studio, vLLM, Groq, OpenRouter — any
//! endpoint Lumi's Python side already speaks.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    pub function: ToolCallFn,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallFn {
    #[serde(default)]
    pub name: String,
    /// Raw JSON arguments, accumulated from stream fragments.
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// Text content. Empty when the assistant message carries only tool calls.
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `reasoning_content` (DeepSeek/Qwen style) — kept for replay, never sent
    /// unless the provider tolerates it, which is why it is flattened out.
    #[serde(skip)]
    pub reasoning: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), ..Default::default() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), ..Default::default() }
    }
    pub fn tool_result(call_id: &str, name: &str, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(call_id.to_string()),
            name: Some(name.to_string()),
            reasoning: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON-schema for `parameters`.
    pub parameters: Value,
}

#[derive(Debug, Default, Clone)]
pub struct StreamUpdate {
    pub content_delta: String,
    pub reasoning_delta: String,
}

#[derive(Debug, Clone, Serialize)]
struct RequestMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: &'a Option<String>,
}

pub struct LlmClient {
    http: Client,
    base_url: String,
    api_key: Option<String>,
}

impl LlmClient {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(15))
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .expect("reqwest client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }

    /// Stream one completion. `on_update` receives every parsed delta; the
    /// final assembled `ChatMessage` (content + tool calls) is returned.
    pub async fn stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolDef],
        on_update: &mut (dyn FnMut(StreamUpdate) + Send),
    ) -> Result<ChatMessage> {
        let url = format!("{}/chat/completions", self.base_url);
        let request_messages: Vec<RequestMessage> = messages
            .iter()
            .map(|m| RequestMessage {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                },
                content: &m.content,
                tool_calls: m.tool_calls.as_ref(),
                tool_call_id: &m.tool_call_id,
                name: &m.name,
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": request_messages,
            "stream": true,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(
                tools
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters,
                            }
                        })
                    })
                    .collect(),
            );
        }
        let mut request = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            // Body text can be huge; keep the first line that mentions anything.
            let short: String = text.lines().take(4).collect::<Vec<_>>().join(" ");
            return Err(anyhow!("{} — {}", status, truncate(&short, 400)));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::<u8>::new();
        let mut assembled = ChatMessage {
            role: Role::Assistant,
            ..Default::default()
        };
        let mut calls: Vec<ToolCall> = Vec::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buffer.extend_from_slice(&bytes);
            while let Some(pos) = find_byte(&buffer, b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data == "[DONE]" {
                    continue;
                }
                let Ok(json) = serde_json::from_str::<Value>(data) else { continue };
                let Some(choices) = json.get("choices").and_then(Value::as_array) else { continue };
                let Some(choice) = choices.first() else { continue };
                let delta = choice.get("delta");
                if let Some(delta) = delta {
                    if let Some(text) = delta.get("content").and_then(Value::as_str) {
                        assembled.content.push_str(text);
                        on_update(StreamUpdate {
                            content_delta: text.to_string(),
                            ..Default::default()
                        });
                    }
                    if let Some(text) = delta
                        .get("reasoning_content")
                        .or_else(|| delta.get("reasoning"))
                        .and_then(Value::as_str)
                    {
                        assembled.reasoning.push_str(text);
                        on_update(StreamUpdate {
                            reasoning_delta: text.to_string(),
                            ..Default::default()
                        });
                    }
                    if let Some(fragments) = delta.get("tool_calls").and_then(Value::as_array) {
                        for frag in fragments {
                            let index = frag.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                            // A broken or hostile stream must not grow the
                            // vector without bound.
                            if index > 128 {
                                continue;
                            }
                            let id = frag.get("id").and_then(Value::as_str).map(str::to_string);
                            let name = frag
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(Value::as_str)
                                .map(str::to_string);
                            let args = frag
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(Value::as_str)
                                .map(str::to_string);
                            while calls.len() <= index {
                                calls.push(ToolCall {
                                    kind: "function".into(),
                                    ..Default::default()
                                });
                            }
                            if let Some(id) = id {
                                calls[index].id = id;
                            }
                            if let Some(name) = name {
                                calls[index].function.name.push_str(&name);
                            }
                            if let Some(args) = args {
                                calls[index].function.arguments.push_str(&args);
                            }
                        }
                    }
                }
            }
        }

        if !calls.is_empty() {
            assembled.tool_calls = Some(calls);
        }
        Ok(assembled)
    }
}

fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{}…", cut)
    }
}

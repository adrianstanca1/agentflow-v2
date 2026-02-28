import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import toast, { Toaster } from "react-hot-toast";
import {
  Bot, Code2, Search, BarChart3, Terminal, Server, Database,
  Zap, Send, Square, Loader2, Plus, Trash2, Copy, Check, X,
  Activity, Settings, Menu, ChevronRight, ChevronDown,
  Download, Upload, RefreshCw, Play, AlertCircle, CheckCircle2,
  Cpu, HardDrive, Package, Layers, GitBranch, Eye, FlaskConical,
  Sparkles, ArrowRight, MessageSquare, Shield,
  FileText, Boxes, Plug, Brain, Wrench, Gauge, Info, Radio, CircleDot,
  Wifi, WifiOff, SortDesc, Filter, Star, Clock, Hash, Tag, ChevronUp,
} from "lucide-react";

// Config
// Automatically use the same host the page is served from — works on any server
const API = (import.meta as any).env?.VITE_API_URL ||
  (typeof window !== "undefined" ? `${window.location.protocol}//${window.location.host}` : "http://localhost:8000");
const WS_URL = (import.meta as any).env?.VITE_WS_URL ||
  (typeof window !== "undefined" ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}` : "ws://localhost:8000");

async function apiFetch(path: string, opts: RequestInit = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

type Page = "chat" | "models" | "agents" | "mcp" | "knowledge" | "openclaw" | "settings";

interface AgentInfo {
  id: string; name: string; description: string; type: string;
  model: string; prefer_local: boolean; tools: string[];
}
interface OllamaModel {
  name: string; size: string; size_bytes: number; family: string; host_url: string;
}
interface PopularModel {
  name: string; category: string; size: string; description: string; tags: string[];
}
interface StreamEvent {
  event_type: string; content: any; agent_id?: string; agent_name?: string;
  session_id?: string; metadata?: Record<string, any>; model_used?: string;
}
interface Message {
  id: string; role: "user" | "agent"; content: string;
  events: StreamEvent[]; agent: string; ts: Date; status: "streaming" | "done" | "error";
  model?: string;
}
interface CloudProvider {
  id: string; name: string; icon: string; color: string; docs: string;
  configured: boolean; healthy: boolean | null; latency_ms: number | null;
  error: string | null; model_count: number;
}
interface UnifiedModel {
  id: string; name: string; source: "ollama" | "cloud"; provider: string;
  provider_name: string; provider_icon: string; provider_color: string;
  category: string; configured: boolean; cost_label: string;
  description?: string; tags?: string[]; is_latest?: boolean;
  context_length?: number; size?: string;
}

const AGENT_META: Record<string, { icon: React.ReactNode; color: string }> = {
  coding:      { icon: <Code2 className="w-4 h-4" />,       color: "from-blue-500 to-cyan-500" },
  research:    { icon: <Search className="w-4 h-4" />,      color: "from-emerald-500 to-green-500" },
  data_analyst:{ icon: <BarChart3 className="w-4 h-4" />,   color: "from-violet-500 to-purple-500" },
  devops:      { icon: <Server className="w-4 h-4" />,      color: "from-orange-500 to-amber-500" },
  writer:      { icon: <FileText className="w-4 h-4" />,    color: "from-pink-500 to-rose-500" },
  sql:         { icon: <Database className="w-4 h-4" />,    color: "from-teal-500 to-cyan-600" },
  qa:          { icon: <FlaskConical className="w-4 h-4" />, color: "from-yellow-500 to-orange-500" },
  assistant:   { icon: <Bot className="w-4 h-4" />,         color: "from-slate-500 to-gray-500" },
};

function cx(...args: (string | boolean | undefined | null)[]) {
  return args.filter(Boolean).join(" ");
}

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
      className="text-neutral-500 hover:text-white transition-colors">
      {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  );
}

function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <div onClick={() => onChange(!on)}
      className={cx("w-9 h-5 rounded-full relative flex items-center cursor-pointer transition-colors flex-shrink-0", on ? "bg-green-500" : "bg-neutral-700")}>
      <div className={cx("w-3.5 h-3.5 rounded-full bg-white absolute shadow transition-transform", on ? "translate-x-[22px]" : "translate-x-[3px]")} />
    </div>
  );
}


// ─── Unified Model Picker ──────────────────────────────────────────────────────
function ModelPicker({ value, onChange, compact = false }: {
  value: string; onChange: (v: string) => void; compact?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<{ local: UnifiedModel[]; cloud: UnifiedModel[] }>({ local: [], cloud: [] });
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<"local" | "cloud">("local");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => { apiFetch("/models/all").then(setModels).catch(() => {}); }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    if (open) document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const all = tab === "local" ? models.local : models.cloud;
  const filtered = all.filter(m =>
    !search ||
    m.name.toLowerCase().includes(search.toLowerCase()) ||
    m.provider_name.toLowerCase().includes(search.toLowerCase())
  );
  const selected = [...models.local, ...models.cloud].find(m => m.id === value);

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(p => !p)}
        className={cx("flex items-center gap-1.5 bg-neutral-900 border border-neutral-800 rounded-lg px-2.5 transition-all hover:border-neutral-600 text-xs", compact ? "py-1" : "py-1.5")}>
        {selected ? (
          <><span>{selected.provider_icon}</span><span className="text-neutral-300 font-mono truncate max-w-[140px]">{selected.name}</span></>
        ) : (
          <span className="text-neutral-600">Select model...</span>
        )}
        <ChevronDown className="w-3 h-3 text-neutral-600 flex-shrink-0" />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -4, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -4, scale: 0.97 }} transition={{ duration: 0.1 }}
            className="absolute bottom-full mb-2 left-0 w-80 bg-neutral-900 border border-neutral-800 rounded-xl shadow-2xl overflow-hidden z-50">
            <div className="p-2 border-b border-neutral-800">
              <input autoFocus value={search} onChange={e => setSearch(e.target.value)} placeholder="Search models..."
                className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-2.5 py-1.5 text-xs text-neutral-300 placeholder-neutral-600 focus:outline-none" />
              <div className="flex mt-2 gap-1">
                {(["local", "cloud"] as const).map(t => (
                  <button key={t} onClick={() => setTab(t)} className={cx("flex-1 py-1 rounded-lg text-xs font-medium capitalize transition-colors", tab === t ? "bg-neutral-700 text-white" : "text-neutral-600 hover:text-neutral-400")}>
                    {t === "local" ? `🦙 Local (${models.local.length})` : `☁️ Cloud (${models.cloud.filter(m => m.configured).length}/${models.cloud.length})`}
                  </button>
                ))}
              </div>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {filtered.length === 0 ? (
                <div className="p-4 text-center text-xs text-neutral-600">
                  {tab === "local" ? "No local models installed yet" : "Add API keys in Settings to unlock cloud models"}
                </div>
              ) : filtered.map(m => (
                <button key={m.id} onClick={() => { onChange(m.id); setOpen(false); setSearch(""); }}
                  className={cx("w-full text-left flex items-center gap-2.5 px-3 py-2 hover:bg-neutral-800 transition-colors", value === m.id && "bg-neutral-800")}>
                  <span className="text-base flex-shrink-0">{m.provider_icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-xs font-mono text-neutral-200 truncate">{m.name}</span>
                      {m.is_latest && <span className="text-[9px] px-1 py-0.5 rounded bg-blue-500/20 text-blue-400">new</span>}
                      {!m.configured && tab === "cloud" && <span className="text-[9px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-500">no key</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-neutral-600">{m.provider_name}</span>
                      {m.cost_label && <span className="text-[10px] text-neutral-700">{m.cost_label}</span>}
                    </div>
                  </div>
                  {value === m.id && <Check className="w-3 h-3 text-green-400 flex-shrink-0" />}
                </button>
              ))}
            </div>
            {tab === "cloud" && models.cloud.filter(m => !m.configured).length > 0 && (
              <div className="p-2 border-t border-neutral-800 text-center">
                <p className="text-[10px] text-neutral-600">Add API keys in <span className="text-neutral-400">Settings → Cloud Providers</span></p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Nav Sidebar
function NavSidebar({ page, setPage, stats }: { page: Page; setPage: (p: Page) => void; stats: any }) {
  const nav: { id: Page; icon: React.ReactNode; label: string }[] = [
    { id: "chat",      icon: <MessageSquare className="w-5 h-5" />, label: "Chat" },
    { id: "models",    icon: <Cpu className="w-5 h-5" />,           label: "Models" },
    { id: "agents",    icon: <Bot className="w-5 h-5" />,           label: "Agents" },
    { id: "mcp",       icon: <Plug className="w-5 h-5" />,          label: "MCP" },
    { id: "knowledge", icon: <Database className="w-5 h-5" />,      label: "Knowledge" },
    { id: "settings",  icon: <Settings className="w-5 h-5" />,      label: "Settings" },
  ];
  const healthy = stats?.ollama?.healthy_hosts > 0;
  return (
    <div className="w-[60px] flex-shrink-0 border-r border-neutral-800 flex flex-col items-center py-4 gap-1 bg-neutral-950">
      <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-blue-600 flex items-center justify-center mb-4">
        <Sparkles className="w-5 h-5 text-white" />
      </div>
      {nav.map(n => (
        <button key={n.id} onClick={() => setPage(n.id)} title={n.label}
          className={cx("w-11 h-11 rounded-xl flex items-center justify-center transition-all",
            page === n.id ? "bg-white/10 text-white" : "text-neutral-600 hover:text-neutral-300 hover:bg-white/5")}>
          {n.icon}
        </button>
      ))}
      <div className="flex-1" />
      <div className="w-2.5 h-2.5 rounded-full" style={{ background: healthy ? "#22c55e" : "#ef4444" }} title={healthy ? "Ollama connected" : "Ollama offline"} />
    </div>
  );
}

// Chat Page
function ChatPage({ agents }: { agents: AgentInfo[] }) {
  const [agentKey, setAgentKey] = useState("assistant");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [model, setModel] = useState("");
  const [provider, setProvider] = useState("");
  const [preferLocal, setPreferLocal] = useState(false);
  const [sideOpen, setSideOpen] = useState(true);
  const [allProviders, setAllProviders] = useState<any[]>([]);
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  const [sessionId] = useState(() => crypto.randomUUID());
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const meta = AGENT_META[agentKey] || AGENT_META.assistant;

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
    // Load configured providers for model picker
    apiFetch("/openclaw/providers").then((data: any[]) => {
      const configured = data.filter(p => p.configured);
      setAllProviders(configured);
      if (configured.length > 0 && !provider) {
        const first = configured[0];
        setProvider(first.id);
        setModel(first.default_model || "");
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!provider) return;
    apiFetch(`/openclaw/models?provider=${provider}`).then((data: any[]) => {
      const names = data.map(m => m.name);
      setProviderModels(names);
      if (names.length > 0 && !model) setModel(names[0]);
    }).catch(() => {});
  }, [provider]);

  const SUGGESTIONS: Record<string, string[]> = {
    assistant:   ["Explain LangGraph vs CrewAI", "Best prompt engineering practices 2026", "Help me plan a microservices architecture"],
    coding:      ["Write a FastAPI app with JWT auth", "Create a React hook for WebSocket data", "Build a Redis rate limiter in Go"],
    research:    ["Research local LLM performance benchmarks", "Leading agentic AI companies in 2026", "Recent papers on RAG optimization"],
    data_analyst:["Analyze a dataset for business insights", "Python script to visualize sales trends", "SQL for finding customer churn"],
    devops:      ["GitHub Actions CI/CD for Python service", "Kubernetes HPA auto-scaling config", "Optimize Dockerfile for faster builds"],
    writer:      ["Write a blog post about AI agents 2026", "Product copy for an AI platform launch", "Technical README for open source project"],
    sql:         ["Query for top customers by lifetime value", "Schema for multi-tenant SaaS app", "Analyze query performance, suggest indexes"],
    qa:          ["Write pytest for a FastAPI auth endpoint", "Playwright e2e tests for a login flow", "Review code for security vulnerabilities"],
  };

  const send = useCallback(async () => {
    if (!input.trim() || running) return;
    const task = input.trim();
    setInput(""); setRunning(true);
    const uid = crypto.randomUUID(), aid = crypto.randomUUID();
    setMessages(m => [...m,
      { id: uid, role: "user", content: task, events: [], agent: agentKey, ts: new Date(), status: "done" },
      { id: aid, role: "agent", content: "", events: [], agent: agentKey, ts: new Date(), status: "streaming" },
    ]);
    try {
      const body: any = { agent_key: agentKey, task, stream: true, prefer_local: preferLocal, session_id: sessionId };
      if (model) body.model_override = model;
      // Also store conversation locally
      setConversations(prev => [...prev.slice(-49), { 
        id: uid, agent: agentKey, task, ts: new Date().toISOString(),
        model: model || "auto", provider: provider || "auto"
      }]);
      const res = await fetch(`${API}/agents/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error(await res.text());
      const reader = res.body!.getReader(); const dec = new TextDecoder();
      let buf = "", finalContent = "", evts: StreamEvent[] = [];
      abortRef.current = () => reader.cancel();
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt: StreamEvent = JSON.parse(line.slice(6)); evts.push(evt);
            // Accumulate stream tokens for real-time display
            if (evt.event_type === "stream_token" && typeof evt.content === "string") {
              finalContent += evt.content;
            } else if ((evt.event_type === "output" || evt.event_type === "complete") && typeof evt.content === "string" && evt.content) {
              finalContent = evt.content; // Use full content from complete event
            }
            setMessages(prev => prev.map(m => m.id === aid ? { 
              ...m, 
              content: finalContent || m.content, 
              events: [...evts],
              status: evt.event_type === "complete" ? "done" : evt.event_type === "error" ? "error" : "streaming",
              model: evt.model_used || m.model,
            } : m));
          } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        toast.error(e.message);
        setMessages(prev => prev.map(m => m.id === aid ? { ...m, content: `Error: ${e.message}`, status: "error" } : m));
      }
    } finally { 
      setRunning(false); 
      abortRef.current = null; 
      inputRef.current?.focus(); 
    }
  }, [input, running, agentKey, model, preferLocal]);

  const agentList = agents.length > 0 ? agents :
    Object.keys(AGENT_META).map(k => ({ id: k, name: k, description: "", type: k, model: "", prefer_local: false, tools: [] }));

  return (
    <div className="flex h-full">
      <AnimatePresence initial={false}>
        {sideOpen && (
          <motion.div initial={{ width: 0 }} animate={{ width: 220 }} exit={{ width: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 320 }}
            className="border-r border-neutral-800 flex flex-col bg-neutral-950 overflow-hidden flex-shrink-0">
            <div className="p-3 border-b border-neutral-800">
              <p className="text-[10px] font-semibold text-neutral-600 uppercase tracking-wider">Select Agent</p>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
              {agentList.map(a => {
                const m = AGENT_META[a.id] || AGENT_META.assistant;
                const sel = agentKey === a.id;
                return (
                  <button key={a.id} onClick={() => setAgentKey(a.id)}
                    className={cx("w-full text-left p-2.5 rounded-lg transition-all flex items-center gap-2.5",
                      sel ? "bg-white/8 border border-white/10" : "hover:bg-white/4 border border-transparent")}>
                    <div className={cx("w-7 h-7 rounded-lg bg-gradient-to-br flex items-center justify-center text-white flex-shrink-0", m.color)}>{m.icon}</div>
                    <div className="min-w-0">
                      <div className="text-xs font-medium text-neutral-200 capitalize">{a.name || a.id}</div>
                      <div className="text-[10px] text-neutral-600 truncate">{a.model || "default"}</div>
                    </div>
                  </button>
                );
              })}
            </div>
            {/* Conversation history */}
            {conversations.length > 0 && (
              <div className="border-t border-neutral-800 px-2 pt-2 pb-1">
                <p className="text-[10px] text-neutral-700 font-semibold uppercase tracking-wider px-1 mb-1">
                  Recent · {conversations.length}
                </p>
                <div className="space-y-0.5 max-h-32 overflow-y-auto">
                  {[...conversations].reverse().slice(0, 10).map((c, i) => (
                    <div key={i}
                      className="text-[10px] text-neutral-600 hover:text-neutral-400 px-2 py-1 rounded hover:bg-white/4 cursor-pointer truncate transition-colors"
                      title={c.task}
                      onClick={() => setInput(c.task)}>
                      {c.task.slice(0, 40)}
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="p-3 border-t border-neutral-800 space-y-2">
              <p className="text-[10px] text-neutral-600 font-medium uppercase tracking-wider">Model Override</p>
              <ModelPicker value={model} onChange={setModel} compact />
              <div className="flex items-center gap-2">
                <Toggle on={preferLocal} onChange={setPreferLocal} />
                <span className="text-[10px] text-neutral-600">Prefer Local</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="h-12 border-b border-neutral-800 flex items-center px-4 gap-3 flex-shrink-0">
          <button onClick={() => setSideOpen(p => !p)} className="text-neutral-600 hover:text-neutral-300 transition-colors">
            <Menu className="w-4 h-4" />
          </button>
          <div className={cx("w-5 h-5 rounded-md bg-gradient-to-br flex items-center justify-center text-white", meta.color)}>
            <div className="scale-75">{meta.icon}</div>
          </div>
          <span className="text-sm text-neutral-300 font-medium capitalize">{agentKey.replace("_", " ")} Agent</span>
          {preferLocal && <span className="text-[10px] px-2 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20">Local</span>}
          <div className="flex-1" />
          {messages.length > 0 && <button onClick={() => setMessages([])} className="text-[11px] text-neutral-600 hover:text-neutral-400 transition-colors">Clear</button>}
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4">
          {messages.length === 0 ? (
            <div className="max-w-xl mx-auto mt-10">
              <div className={cx("w-12 h-12 mx-auto mb-3 rounded-2xl bg-gradient-to-br flex items-center justify-center text-white", meta.color)}>
                <div className="scale-125">{meta.icon}</div>
              </div>
              <p className="text-center text-neutral-500 text-sm mb-6 capitalize">{agentKey.replace("_", " ")} — ready</p>
              <div className="space-y-2">
                {(SUGGESTIONS[agentKey] || SUGGESTIONS.assistant).map((s, i) => (
                  <motion.button key={i} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}
                    onClick={() => setInput(s)}
                    className="w-full text-left p-3 rounded-xl border border-neutral-800 bg-neutral-900/40 hover:bg-neutral-900 hover:border-neutral-700 transition-all group">
                    <div className="flex items-center gap-2.5">
                      <ArrowRight className="w-3.5 h-3.5 text-neutral-700 group-hover:text-neutral-400 transition-colors flex-shrink-0" />
                      <p className="text-sm text-neutral-500 group-hover:text-neutral-300 transition-colors">{s}</p>
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-2xl mx-auto space-y-4">
              {messages.map(msg => <ChatMessage key={msg.id} msg={msg} />)}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Model picker panel */}
        {showModelPicker && (
          <div className="border-t border-neutral-800/60 bg-[#0a0a0c] px-4 py-3">
            <div className="max-w-2xl mx-auto flex items-center gap-3">
              <div className="flex-1">
                <div className="text-[10px] text-neutral-600 mb-1 uppercase tracking-wider">Provider</div>
                <select value={provider} onChange={e => { setProvider(e.target.value); setModel(""); }}
                  className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-2.5 py-1.5 text-xs text-neutral-300 focus:outline-none focus:border-cyan-600">
                  {allProviders.length > 0
                    ? allProviders.map((p: any) => <option key={p.id} value={p.id}>{p.icon} {p.name}</option>)
                    : <option value="">Auto (no providers configured)</option>
                  }
                </select>
              </div>
              <div className="flex-1">
                <div className="text-[10px] text-neutral-600 mb-1 uppercase tracking-wider">Model</div>
                <select value={model} onChange={e => setModel(e.target.value)}
                  className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-2.5 py-1.5 text-xs text-neutral-300 focus:outline-none focus:border-cyan-600">
                  {providerModels.length > 0
                    ? providerModels.map(m => <option key={m} value={m}>{m}</option>)
                    : model ? <option value={model}>{model}</option> : <option value="">Default model</option>
                  }
                </select>
              </div>
              <button onClick={() => setShowModelPicker(false)}
                className="text-neutral-700 hover:text-neutral-400 text-base mt-4 flex-shrink-0">✕</button>
            </div>
          </div>
        )}

        {/* Input area */}
        <div className="border-t border-neutral-800 p-3">
          <div className="max-w-2xl mx-auto">
            <div className="flex gap-2 items-end bg-neutral-900 border border-neutral-700 rounded-xl px-3 py-2.5 focus-within:border-neutral-500 transition-colors">
              <textarea ref={inputRef} value={input} rows={1}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder={`Message ${agentKey.replace("_", " ")} agent…`}
                className="flex-1 bg-transparent text-sm text-neutral-200 placeholder-neutral-600 resize-none focus:outline-none leading-relaxed"
                style={{ minHeight: "22px", maxHeight: "160px" }}
                onInput={e => { const t = e.currentTarget; t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; }} />
              {running ? (
                <button onClick={() => { abortRef.current?.(); setRunning(false); }}
                  className="w-8 h-8 rounded-lg bg-red-500/20 border border-red-500/30 flex items-center justify-center text-red-400 hover:bg-red-500/30 transition-colors flex-shrink-0">
                  <Square className="w-3.5 h-3.5" />
                </button>
              ) : (
                <button onClick={send} disabled={!input.trim()}
                  className={cx("w-8 h-8 rounded-lg flex items-center justify-center transition-all flex-shrink-0",
                    input.trim() ? cx("bg-gradient-to-br text-white shadow-lg", meta.color) : "bg-neutral-800 text-neutral-600 cursor-not-allowed")}>
                  <Send className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
            <div className="flex items-center justify-between mt-1.5 px-0.5">
              <button onClick={() => setShowModelPicker(p => !p)}
                className={cx("flex items-center gap-1 text-[10px] transition-colors",
                  showModelPicker ? "text-cyan-500" : "text-neutral-700 hover:text-neutral-500")}>
                <span>⚙</span>
                <span className="font-mono">
                  {model
                    ? (model.split("/").pop()?.split(":")[0] || model)
                    : "auto"}
                </span>
                {provider && allProviders.find((p: any) => p.id === provider) && (
                  <span className="text-neutral-800">· {allProviders.find((p: any) => p.id === provider)?.name}</span>
                )}
              </button>
              <p className="text-[10px] text-neutral-800">↵ send · ⇧↵ newline</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatMessage({ msg }: { msg: Message }) {
  const [expanded, setExpanded] = useState(false);
  const meta = AGENT_META[msg.agent] || AGENT_META.assistant;
  const toolCalls = msg.events.filter(e => e.event_type === "tool_call");
  const thoughts = msg.events.filter(e => e.event_type === "thought");

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-neutral-800 border border-neutral-700 rounded-2xl rounded-tr-sm px-4 py-2.5">
          <p className="text-sm text-neutral-200 leading-relaxed whitespace-pre-wrap">{msg.content}</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3 group">
      <div className={cx("w-8 h-8 rounded-xl bg-gradient-to-br flex items-center justify-center text-white flex-shrink-0 mt-0.5", meta.color)}>
        {meta.icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className="text-xs font-semibold text-neutral-400 capitalize">{msg.agent.replace("_", " ")}</span>
          {(msg as any).model && (
            <span className="text-[10px] font-mono text-cyan-700 bg-cyan-950/40 border border-cyan-900/40 px-1.5 py-0.5 rounded-full">
              {(msg as any).model.split("/").pop()?.split(":")[0] || (msg as any).model}
            </span>
          )}
          <span className="text-[10px] text-neutral-800">{msg.ts.toLocaleTimeString()}</span>
          {msg.status === "streaming" && (
            <span className="flex items-center gap-1 text-[10px] text-neutral-500">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />streaming
            </span>
          )}
          {msg.status === "error" && <span className="text-[10px] text-red-500">⚠ error</span>}
          {msg.status === "done" && msg.content && (
            <button onClick={() => navigator.clipboard.writeText(msg.content)}
              className="ml-auto opacity-0 group-hover:opacity-100 text-[10px] text-neutral-700 hover:text-neutral-400 transition-all">
              copy
            </button>
          )}
        </div>
        {(toolCalls.length > 0 || thoughts.length > 0) && (
          <div className="mb-2">
            <button onClick={() => setExpanded(p => !p)}
              className="flex items-center gap-1 text-[10px] text-neutral-600 hover:text-neutral-400 transition-colors mb-1">
              {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              {toolCalls.length > 0 && `${toolCalls.length} tool call${toolCalls.length > 1 ? "s" : ""}`}
              {thoughts.length > 0 && ` · ${thoughts.length} thought${thoughts.length > 1 ? "s" : ""}`}
            </button>
            <AnimatePresence>
              {expanded && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden pl-2 border-l border-neutral-800 space-y-1 mb-2">
                  {thoughts.map((t, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <Brain className="w-3 h-3 text-purple-500 mt-0.5 flex-shrink-0" />
                      <p className="text-[11px] text-neutral-600">{String(t.content).slice(0, 120)}</p>
                    </div>
                  ))}
                  {toolCalls.map((tc, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <Terminal className="w-3 h-3 text-cyan-500 flex-shrink-0" />
                      <span className="text-[11px] font-mono text-cyan-400">{tc.content?.name || "tool"}</span>
                      <span className="text-[10px] text-neutral-700 truncate">{JSON.stringify(tc.content?.args || {}).slice(0, 60)}</span>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
        {msg.content ? (
          <div className="text-sm text-neutral-300 leading-relaxed">
            <ReactMarkdown components={{
              code({ inline, className, children, ...props }: any) {
                const lang = /language-(\w+)/.exec(className || "")?.[1];
                return !inline && lang ? (
                  <div className="my-2">
                    <div className="flex items-center justify-between bg-neutral-950 border border-neutral-800 rounded-t-lg px-3 py-1">
                      <span className="text-[10px] text-neutral-600 font-mono">{lang}</span>
                      <CopyBtn text={String(children)} />
                    </div>
                    <SyntaxHighlighter style={atomDark} language={lang} PreTag="div"
                      className="!rounded-t-none !rounded-b-lg !m-0 !text-xs !border-x !border-b !border-neutral-800">
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  </div>
                ) : <code className="bg-neutral-800 text-cyan-300 rounded px-1 py-0.5 text-xs font-mono" {...props}>{children}</code>;
              },
              p: ({ children }) => <p className="mb-2 last:mb-0 text-neutral-300">{children}</p>,
              h1: ({ children }) => <h1 className="text-base font-bold text-white mt-3 mb-1.5">{children}</h1>,
              h2: ({ children }) => <h2 className="text-sm font-semibold text-white mt-2.5 mb-1">{children}</h2>,
              h3: ({ children }) => <h3 className="text-sm font-medium text-neutral-200 mt-2 mb-1">{children}</h3>,
              ul: ({ children }) => <ul className="pl-4 space-y-0.5 mb-2 list-disc">{children}</ul>,
              ol: ({ children }) => <ol className="pl-4 space-y-0.5 mb-2 list-decimal">{children}</ol>,
              li: ({ children }) => <li className="text-neutral-400 text-sm">{children}</li>,
              a: ({ children, href }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">{children}</a>,
              strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
              blockquote: ({ children }) => <blockquote className="border-l-2 border-neutral-700 pl-3 text-neutral-500 italic my-2">{children}</blockquote>,
              table: ({ children }) => <div className="overflow-x-auto my-2"><table className="border-collapse w-full text-xs">{children}</table></div>,
              th: ({ children }) => <th className="border border-neutral-700 px-2 py-1 bg-neutral-800 text-neutral-200 text-left">{children}</th>,
              td: ({ children }) => <td className="border border-neutral-800 px-2 py-1 text-neutral-400">{children}</td>,
            }}>
              {msg.content}
            </ReactMarkdown>
          </div>
        ) : msg.status === "streaming" ? (
          <div className="flex items-center gap-1.5">
            {[0,1,2].map(i => <div key={i} className="w-1.5 h-1.5 rounded-full bg-neutral-600 animate-bounce" style={{ animationDelay: `${i * 150}ms` }} />)}
          </div>
        ) : null}
      </div>
    </motion.div>
  );
}

// Models Page
function ModelsPage() {
  const [tab, setTab] = useState<"installed" | "hub" | "hosts" | "cloud">("installed");
  const [providers, setProviders] = useState<CloudProvider[]>([]);
  const [cloudModels, setCloudModels] = useState<any[]>([]);
  const [cloudFilter, setCloudFilter] = useState("all");
  const [checking, setChecking] = useState<string | null>(null);
  const [installed, setInstalled] = useState<OllamaModel[]>([]);
  const [popular, setPopular] = useState<PopularModel[]>([]);
  const [hosts, setHosts] = useState<any[]>([]);
  const [running, setRunning] = useState<any[]>([]);
  const [pulling, setPulling] = useState<Record<string, { percent: number; status: string }>>({});
  const [filterCat, setFilterCat] = useState("all");
  const [search, setSearch] = useState("");
  const [newUrl, setNewUrl] = useState(""); const [newName, setNewName] = useState(""); const [newKey, setNewKey] = useState("");

  const load = useCallback(async () => {
    try {
      const [m, p, s, r, cp, cm] = await Promise.all([
        apiFetch("/ollama/models"), apiFetch("/ollama/models/popular"),
        apiFetch("/ollama/hosts"), apiFetch("/ollama/models/running"),
        apiFetch("/cloud/providers"), apiFetch("/cloud/models"),
      ]);
      setInstalled(m); setPopular(p); setHosts(s.hosts || []); setRunning(r);
      setProviders(cp); setCloudModels(cm);
    } catch {}
  }, []);
  useEffect(() => { load(); }, [load]);

  const pullModel = async (name: string) => {
    setPulling(p => ({ ...p, [name]: { percent: 0, status: "starting" } }));
    try {
      const res = await fetch(`${API}/ollama/models/pull`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_name: name }) });
      const reader = res.body!.getReader(); const dec = new TextDecoder(); let buf = "";
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const p = JSON.parse(line.slice(6));
            setPulling(prev => ({ ...prev, [name]: { percent: p.percent || 0, status: p.status || "" } }));
            if (p.status === "success") { load(); setPulling(prev => { const n = { ...prev }; delete n[name]; return n; }); }
          } catch {}
        }
      }
    } catch (e: any) { toast.error(`Pull failed: ${e.message}`); setPulling(prev => { const n = { ...prev }; delete n[name]; return n; }); }
  };

  const del = async (name: string, host: string) => {
    if (!confirm(`Delete ${name}?`)) return;
    await apiFetch(`/ollama/models/${encodeURIComponent(name)}?host_url=${encodeURIComponent(host)}`, { method: "DELETE" });
    toast.success("Deleted"); load();
  };

  const addHost = async () => {
    if (!newUrl) return;
    try {
      const r = await apiFetch("/ollama/hosts", { method: "POST", body: JSON.stringify({ url: newUrl, name: newName || newUrl, host_type: "remote", api_key: newKey || undefined }) });
      toast.success(r.healthy ? "Connected!" : "Added (offline)");
      setNewUrl(""); setNewName(""); setNewKey(""); load();
    } catch (e: any) { toast.error(e.message); }
  };

  const cats = ["all", ...Array.from(new Set(popular.map(m => m.category)))];
  const filtered = popular.filter(m => (filterCat === "all" || m.category === filterCat) && (!search || m.name.toLowerCase().includes(search.toLowerCase())));
  const isInstalled = (name: string) => installed.some(m => m.name === name || m.name.startsWith(name.split(":")[0]));
  const catColors: Record<string, string> = { coding: "text-blue-400 bg-blue-400/10", general: "text-green-400 bg-green-400/10", reasoning: "text-purple-400 bg-purple-400/10", embedding: "text-amber-400 bg-amber-400/10", vision: "text-pink-400 bg-pink-400/10" };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="border-b border-neutral-800 px-6 py-4 flex items-center justify-between flex-shrink-0">
        <div>
          <h2 className="text-sm font-semibold text-white">Model Hub</h2>
          <p className="text-xs text-neutral-500">{installed.length} installed · {hosts.filter((h: any) => h.is_healthy).length} host{hosts.filter((h: any) => h.is_healthy).length !== 1 ? "s" : ""} online</p>
        </div>
        <div className="flex gap-1 bg-neutral-900 border border-neutral-800 rounded-xl p-1">
          {(["installed", "hub", "hosts", "cloud"] as const).map(t => (
            <button key={t} onClick={() => setTab(t as any)} className={cx("px-3 py-1.5 rounded-lg text-xs font-medium transition-colors capitalize", tab === t ? "bg-neutral-700 text-white" : "text-neutral-500 hover:text-neutral-300")}>{t === "cloud" ? "☁️ Cloud" : t}</button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {tab === "installed" && (
          <div>
            {running.length > 0 && (
              <div className="mb-4 p-3 rounded-xl bg-green-500/5 border border-green-500/20">
                <p className="text-xs text-green-400 font-medium mb-2 flex items-center gap-1.5"><Radio className="w-3 h-3" /> Running</p>
                <div className="flex flex-wrap gap-1.5">
                  {running.map((m: any, i: number) => <span key={i} className="text-xs px-2 py-1 rounded bg-green-500/10 text-green-300 border border-green-500/20 flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />{m.name}</span>)}
                </div>
              </div>
            )}
            {installed.length === 0 ? (
              <div className="text-center py-16"><HardDrive className="w-10 h-10 text-neutral-700 mx-auto mb-3" /><p className="text-neutral-500 text-sm">No models installed</p><p className="text-neutral-600 text-xs mt-1">Go to Hub tab to pull models</p></div>
            ) : (
              <div className="space-y-2">
                {installed.map((m, i) => (
                  <div key={i} className="flex items-center gap-4 p-3 rounded-xl bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-all group">
                    <Cpu className="w-4 h-4 text-neutral-600 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-mono font-medium text-neutral-200">{m.name}</p>
                      <p className="text-xs text-neutral-600">{m.family} · {m.size} · {m.host_url.replace(/https?:\/\//, "")}</p>
                    </div>
                    <button onClick={() => del(m.name, m.host_url)} className="opacity-0 group-hover:opacity-100 text-neutral-700 hover:text-red-400 p-1 transition-all"><Trash2 className="w-3.5 h-3.5" /></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "hub" && (
          <div>
            <div className="flex gap-3 mb-4">
              <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search models..."
                className="flex-1 bg-neutral-900 border border-neutral-800 rounded-xl px-3 py-2 text-sm text-neutral-300 placeholder-neutral-700 focus:outline-none focus:border-neutral-600" />
              <div className="flex gap-1 bg-neutral-900 border border-neutral-800 rounded-xl p-1">
                {cats.map(c => <button key={c} onClick={() => setFilterCat(c)} className={cx("px-2.5 py-1 rounded-lg text-xs capitalize transition-colors", filterCat === c ? "bg-neutral-700 text-white" : "text-neutral-600 hover:text-neutral-400")}>{c}</button>)}
              </div>
            </div>
            <div className="space-y-2">
              {filtered.map((m, i) => {
                const inst = isInstalled(m.name); const ps = pulling[m.name];
                return (
                  <div key={i} className="p-4 rounded-xl bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-all">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <span className="text-sm font-mono font-medium text-neutral-100">{m.name}</span>
                          <span className={cx("text-[10px] px-1.5 py-0.5 rounded capitalize font-medium", catColors[m.category] || "text-neutral-500 bg-neutral-800")}>{m.category}</span>
                          <span className="text-[10px] text-neutral-600">{m.size}</span>
                          {inst && <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20">Installed</span>}
                        </div>
                        <p className="text-xs text-neutral-500 mb-1.5">{m.description}</p>
                        <div className="flex flex-wrap gap-1">{m.tags.map(t => <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-600">#{t}</span>)}</div>
                      </div>
                      <div className="flex-shrink-0 pt-1">
                        {ps ? (
                          <div className="flex items-center gap-2 w-28">
                            <div className="flex-1 h-1 bg-neutral-800 rounded-full overflow-hidden"><div className="h-full bg-blue-500 transition-all rounded-full" style={{ width: `${ps.percent}%` }} /></div>
                            <span className="text-[10px] text-neutral-500 w-8 text-right">{ps.percent.toFixed(0)}%</span>
                          </div>
                        ) : (
                          <button onClick={() => pullModel(m.name)} disabled={inst}
                            className={cx("flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all", inst ? "bg-neutral-800 text-neutral-600 cursor-default" : "bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20")}>
                            <Download className="w-3 h-3" />{inst ? "Installed" : "Pull"}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}


        {tab === "cloud" && (
          <div className="space-y-6">
            {/* Provider cards */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Cloud Providers</p>
                <button onClick={async () => { setChecking("all"); await apiFetch("/cloud/providers/check", { method: "POST" }); load(); setChecking(null); }}
                  disabled={checking === "all"}
                  className="flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-300 transition-colors">
                  {checking === "all" ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                  Check all
                </button>
              </div>
              <div className="grid grid-cols-2 gap-3 mb-6">
                {providers.map(p => (
                  <div key={p.id} className="p-3.5 rounded-xl bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-all">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xl">{p.icon}</span>
                        <div>
                          <p className="text-sm font-medium text-neutral-200">{p.name}</p>
                          <div className="flex items-center gap-1.5">
                            {p.configured ? (
                              p.healthy === null ? (
                                <span className="text-[10px] text-neutral-600">Not checked</span>
                              ) : p.healthy ? (
                                <span className="flex items-center gap-1 text-[10px] text-green-400"><span className="w-1.5 h-1.5 rounded-full bg-green-400" /> {p.latency_ms?.toFixed(0)}ms</span>
                              ) : (
                                <span className="flex items-center gap-1 text-[10px] text-red-400"><span className="w-1.5 h-1.5 rounded-full bg-red-500" /> Invalid key</span>
                              )
                            ) : (
                              <span className="text-[10px] text-amber-500">No API key</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <button onClick={async () => { setChecking(p.id); await apiFetch(`/cloud/providers/${p.id}/check`, { method: "POST" }); load(); setChecking(null); }}
                        disabled={!p.configured || checking === p.id}
                        className="text-neutral-700 hover:text-neutral-400 transition-colors disabled:opacity-30">
                        {checking === p.id ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                      </button>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-neutral-600">{p.model_count} models</span>
                      {p.docs && <a href={p.docs} target="_blank" rel="noopener noreferrer" className="text-[10px] text-blue-500 hover:text-blue-400">Docs →</a>}
                    </div>
                    {!p.configured && (
                      <div className="mt-2 text-[10px] text-neutral-600 font-mono">Set {p.id.toUpperCase()}_API_KEY in .env</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Cloud model catalog */}
            <div>
              <div className="flex items-center gap-3 mb-3">
                <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider flex-1">Model Catalog</p>
                <div className="flex gap-1 bg-neutral-900 border border-neutral-800 rounded-xl p-1">
                  {["all", ...Array.from(new Set(cloudModels.map((m: any) => m.provider))).slice(0,5)].map((f: any) => (
                    <button key={f} onClick={() => setCloudFilter(f)} className={cx("px-2 py-1 rounded-lg text-[10px] transition-colors capitalize", cloudFilter === f ? "bg-neutral-700 text-white" : "text-neutral-600 hover:text-neutral-400")}>
                      {f === "all" ? "All" : (providers.find(p => p.id === f)?.icon || "") + " " + f}
                    </button>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                {cloudModels.filter((m: any) => cloudFilter === "all" || m.provider === cloudFilter).map((m: any, i: number) => (
                  <div key={i} className={cx("p-3.5 rounded-xl border transition-all",
                    m.configured ? "bg-neutral-900 border-neutral-800 hover:border-neutral-700" : "bg-neutral-950 border-neutral-900 opacity-60")}>
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-0.5">
                          <span className="text-sm font-mono font-medium text-neutral-200">{m.name}</span>
                          {m.is_latest && <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 font-medium">latest</span>}
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-500 capitalize">{m.category}</span>
                          {!m.configured && <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-600 border border-amber-500/20">needs key</span>}
                        </div>
                        <div className="flex items-center gap-3 text-[10px]">
                          <span className="text-neutral-600">{m.provider_icon} {m.provider_name}</span>
                          {m.context_length && <span className="text-neutral-700">{(m.context_length/1000).toFixed(0)}K ctx</span>}
                          <span className="text-neutral-700">{m.cost_label}</span>
                        </div>
                        {m.description && <p className="text-[11px] text-neutral-600 mt-0.5 truncate">{m.description}</p>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab === "hosts" && (
          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
              <p className="text-sm font-medium text-neutral-300 mb-3">Add Ollama Host</p>
              <div className="space-y-2 mb-3">
                <input value={newUrl} onChange={e => setNewUrl(e.target.value)} placeholder="http://192.168.1.100:11434" className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none" />
                <div className="grid grid-cols-2 gap-2">
                  <input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Name (optional)" className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none" />
                  <input value={newKey} onChange={e => setNewKey(e.target.value)} placeholder="API key" type="password" className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none" />
                </div>
              </div>
              <button onClick={addHost} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm text-white transition-colors"><Plus className="w-3.5 h-3.5" /> Connect</button>
            </div>
            <div className="space-y-2">
              {hosts.map((h: any, i: number) => (
                <div key={i} className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
                  <div className="flex items-center gap-3">
                    <div className={cx("w-2.5 h-2.5 rounded-full flex-shrink-0", h.is_healthy ? "bg-green-400" : "bg-red-500")} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-neutral-200">{h.name}</p>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-500 capitalize">{h.type}</span>
                        {h.gpu_count > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400">GPU×{h.gpu_count}</span>}
                      </div>
                      <p className="text-xs text-neutral-600 font-mono">{h.url}</p>
                    </div>
                    <div className="text-right"><p className="text-xs text-neutral-400">{h.model_count} models</p>{h.is_healthy && <p className="text-[10px] text-neutral-600">{h.latency_ms?.toFixed(0)}ms</p>}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Agents Page
function AgentsPage({ agents }: { agents: AgentInfo[] }) {
  const [selected, setSelected] = useState<AgentInfo | null>(null);
  const [switchMod, setSwitchMod] = useState(""); const [local, setLocal] = useState(false);
  const doSwitch = async () => {
    if (!switchMod || !selected) return;
    try { await apiFetch("/agents/switch-model", { method: "POST", body: JSON.stringify({ model: switchMod, agent_key: selected.id, prefer_local: local }) }); toast.success(`Switched to ${switchMod}`); }
    catch (e: any) { toast.error(e.message); }
  };
  return (
    <div className="flex h-full">
      <div className="w-60 border-r border-neutral-800 flex flex-col">
        <div className="p-4 border-b border-neutral-800"><h2 className="text-sm font-semibold text-white">Active Agents</h2><p className="text-xs text-neutral-600">{agents.length} loaded</p></div>
        <div className="flex-1 overflow-y-auto p-2">
          {agents.map(a => {
            const m = AGENT_META[a.id] || AGENT_META.assistant;
            return (
              <button key={a.id} onClick={() => setSelected(a)} className={cx("w-full text-left p-3 rounded-xl mb-1 transition-all flex items-center gap-3", selected?.id === a.id ? "bg-white/8 border border-white/10" : "hover:bg-white/4 border border-transparent")}>
                <div className={cx("w-8 h-8 rounded-xl bg-gradient-to-br flex items-center justify-center text-white flex-shrink-0", m.color)}>{m.icon}</div>
                <div className="min-w-0"><p className="text-xs font-medium text-neutral-200 capitalize">{a.name}</p><p className="text-[10px] text-neutral-600 font-mono truncate w-36">{a.model}</p></div>
              </button>
            );
          })}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-6">
        {selected ? (
          <div className="max-w-lg space-y-4">
            <div className="flex items-center gap-4">
              <div className={cx("w-14 h-14 rounded-2xl bg-gradient-to-br flex items-center justify-center text-white", (AGENT_META[selected.id] || AGENT_META.assistant).color)}>
                <div className="scale-125">{(AGENT_META[selected.id] || AGENT_META.assistant).icon}</div>
              </div>
              <div><h3 className="text-lg font-semibold text-white capitalize">{selected.name}</h3><p className="text-sm text-neutral-500">{selected.description}</p></div>
            </div>
            <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
              <p className="text-xs text-neutral-500 mb-1.5 font-medium uppercase tracking-wider">Current Model</p>
              <code className="text-sm text-blue-300 font-mono">{selected.model}</code>
              <p className="text-xs text-neutral-600 mt-1">{selected.prefer_local ? "Local (Ollama)" : "Via LiteLLM"}</p>
            </div>
            <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
              <p className="text-xs text-neutral-500 mb-2 font-medium uppercase tracking-wider">Tools ({selected.tools.length})</p>
              <div className="flex flex-wrap gap-1.5">{selected.tools.map(t => <span key={t} className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 font-mono">{t}</span>)}</div>
            </div>
            <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
              <p className="text-xs text-neutral-500 mb-3 font-medium uppercase tracking-wider">Switch Model</p>
              <input value={switchMod} onChange={e => setSwitchMod(e.target.value)} placeholder="e.g. qwen2.5-coder:7b or claude-sonnet-4-6"
                className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none mb-2" />
              <div className="flex items-center gap-2 mb-3"><Toggle on={local} onChange={setLocal} /><span className="text-xs text-neutral-500">Use Ollama (local)</span></div>
              <button onClick={doSwitch} className="px-4 py-2 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm text-white transition-colors">Apply</button>
            </div>
          </div>
        ) : (
          <div className="text-center py-16"><Bot className="w-10 h-10 text-neutral-700 mx-auto mb-3" /><p className="text-neutral-500 text-sm">Select an agent</p></div>
        )}
      </div>
    </div>
  );
}

// MCP Page
function MCPPage() {
  const [servers, setServers] = useState<any[]>([]);
  const [popular, setPopular] = useState<any[]>([]);
  const [tools, setTools] = useState<any[]>([]);
  const load = useCallback(async () => {
    try { const [s, p, t] = await Promise.all([apiFetch("/mcp/servers"), apiFetch("/mcp/servers/popular"), apiFetch("/mcp/tools")]); setServers(s); setPopular(p); setTools(t); } catch {}
  }, []);
  useEffect(() => { load(); }, [load]);
  const add = async (srv: any) => {
    try { await apiFetch("/mcp/servers", { method: "POST", body: JSON.stringify({ name: srv.name, transport: srv.transport, command: srv.command, args: srv.args || [], url: srv.url }) }); toast.success(`${srv.name} connected`); load(); }
    catch (e: any) { toast.error(e.message); }
  };
  const catClr: Record<string, string> = { system: "text-blue-400", dev: "text-cyan-400", data: "text-green-400", search: "text-yellow-400", browser: "text-orange-400", comms: "text-pink-400", storage: "text-purple-400", reasoning: "text-violet-400" };
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div><h2 className="text-sm font-semibold text-white">MCP Servers</h2><p className="text-xs text-neutral-500">Model Context Protocol — external tools for your agents</p></div>
          <div className="flex items-center gap-3"><span className="text-xs text-neutral-600">{servers.filter((s: any) => s.status === "connected").length}/{servers.length} connected</span><span className="text-xs text-neutral-600">{tools.length} tools</span></div>
        </div>
        {servers.length > 0 && (
          <div>
            <p className="text-xs font-medium text-neutral-500 mb-2 uppercase tracking-wider">Active</p>
            <div className="space-y-2">
              {servers.map((s: any) => (
                <div key={s.id} className="p-3 rounded-xl bg-neutral-900 border border-neutral-800 flex items-center gap-3">
                  <div className={cx("w-2 h-2 rounded-full flex-shrink-0", s.status === "connected" ? "bg-green-400" : "bg-neutral-600")} />
                  <div className="flex-1 min-w-0"><p className="text-sm text-neutral-200 font-medium">{s.name}</p><p className="text-xs text-neutral-600">{s.tool_count} tools · {s.transport}</p></div>
                  <div className="flex gap-1">{(s.tools || []).slice(0, 3).map((t: any) => <span key={t.name} className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-600 font-mono">{t.name}</span>)}{s.tool_count > 3 && <span className="text-[10px] text-neutral-700">+{s.tool_count - 3}</span>}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        <div>
          <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Available</p>
          <div className="grid grid-cols-2 gap-3">
            {popular.map((s: any, i: number) => (
              <div key={i} className="p-4 rounded-xl bg-neutral-900 border border-neutral-800 hover:border-neutral-700 transition-all">
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2"><span className="text-xl">{s.icon}</span><div><p className="text-sm font-medium text-neutral-200">{s.name}</p><span className={cx("text-[10px] capitalize", catClr[s.category] || "text-neutral-500")}>{s.category}</span></div></div>
                  <button onClick={() => add(s)} className="text-[11px] px-2.5 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition-colors whitespace-nowrap">+ Add</button>
                </div>
                <p className="text-xs text-neutral-600">{s.description}</p>
                {s.env_keys && <p className="text-[10px] text-amber-600 mt-1.5">Needs: {s.env_keys.join(", ")}</p>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Knowledge Page
function KnowledgePage() {
  const [query, setQuery] = useState(""); const [collection, setCollection] = useState("research");
  const [results, setResults] = useState<any[]>([]); const [ingestText, setIngestText] = useState(""); const [searching, setSearching] = useState(false);
  const search = async () => {
    if (!query) return; setSearching(true);
    try { const r = await apiFetch(`/knowledge/search?query=${encodeURIComponent(query)}&collection=${collection}&limit=8`); setResults(Array.isArray(r) ? r : []); }
    catch (e: any) { toast.error(e.message); } finally { setSearching(false); }
  };
  const ingest = async () => {
    if (!ingestText.trim()) return;
    try { await apiFetch("/knowledge/ingest", { method: "POST", body: JSON.stringify({ content: ingestText, collection }) }); toast.success("Ingested!"); setIngestText(""); }
    catch (e: any) { toast.error(e.message); }
  };
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        <div><h2 className="text-sm font-semibold text-white">Knowledge Base</h2><p className="text-xs text-neutral-500">Vector search — Qdrant + Ollama embeddings</p></div>
        <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
          <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Search</p>
          <div className="flex gap-2 mb-2">
            <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === "Enter" && search()} placeholder="Semantic search..."
              className="flex-1 bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none" />
            <select value={collection} onChange={e => setCollection(e.target.value)} className="bg-neutral-800 border border-neutral-700 rounded-lg px-2 py-2 text-sm text-neutral-400 focus:outline-none">
              {["research","coding","docs","general"].map(c => <option key={c}>{c}</option>)}
            </select>
            <button onClick={search} disabled={searching} className="px-4 py-2 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm text-white transition-colors flex items-center gap-2">
              {searching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
            </button>
          </div>
          {results.length > 0 && <div className="space-y-2 mt-3">{results.map((r, i) => (
            <div key={i} className="p-3 rounded-lg bg-neutral-800 border border-neutral-700">
              <div className="flex items-center gap-2 mb-1"><span className="text-xs font-mono text-green-400">{(r.score * 100).toFixed(0)}%</span>{r.payload?.source && <span className="text-xs text-neutral-600">{r.payload.source}</span>}</div>
              <p className="text-xs text-neutral-400">{r.payload?.content?.slice(0, 200)}...</p>
            </div>
          ))}</div>}
        </div>
        <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
          <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Ingest Document</p>
          <textarea value={ingestText} onChange={e => setIngestText(e.target.value)} placeholder="Paste document text..." rows={4}
            className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2.5 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none resize-none mb-2" />
          <button onClick={ingest} className="px-4 py-2 rounded-lg bg-blue-500/20 border border-blue-500/30 text-blue-400 text-sm hover:bg-blue-500/30 transition-colors flex items-center gap-2"><Upload className="w-3.5 h-3.5" /> Ingest</button>
        </div>
      </div>
    </div>
  );
}

// Settings Page


// ─── OpenClaw Page ─────────────────────────────────────────────────────────────
function OpenClawPage() {
  const [provider, setProvider] = useState("ollama");
  const [model, setModel] = useState("");
  const [cwd, setCwd] = useState(".");
  const [sessionId] = useState(() => "oc-" + Math.random().toString(36).slice(2, 8));
  const [task, setTask] = useState("");
  const [events, setEvents] = useState<Array<{type: string; content: any; ts: number}>>([]);
  const [running, setRunning] = useState(false);
  const [providers, setProviders] = useState<Array<{id: string; name: string; icon: string; configured: boolean; free: boolean; default_model: string}>>([]);
  const [models, setModels] = useState<Array<{name: string; installed: boolean; provider: string}>>([]);
  const [files, setFiles] = useState<Array<{name: string; path: string; is_dir: boolean; size: number}>>([]);
  const [openFile, setOpenFile] = useState<{path: string; content: string; extension: string; lines?: number} | null>(null);
  const [activeTab, setActiveTab] = useState<"agent" | "files" | "editor">("agent");
  const [checkResult, setCheckResult] = useState<{ok: boolean; latency_ms?: number; error?: string} | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    apiFetch("/openclaw/providers").then((data: any[]) => {
      setProviders(data);
      // Auto-select first configured provider
      const first = data.find(p => p.configured) || data[0];
      if (first) {
        setProvider(first.id);
        setModel(first.default_model || "");
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!provider) return;
    apiFetch(`/openclaw/models?provider=${provider}`).then((data: any[]) => {
      setModels(data);
      if (data.length > 0 && !model) setModel(data[0].name);
    }).catch(() => {});
  }, [provider]);

  const testProvider = async () => {
    setCheckResult(null);
    const r = await apiFetch(`/openclaw/providers/${provider}/check`, { method: "POST" }).catch(() => ({ok: false, error: "Request failed"}));
    setCheckResult(r);
  };

  useEffect(() => {
    loadFiles(cwd);
  }, [cwd]);

  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const loadFiles = async (path: string) => {
    try {
      const data = await apiFetch(`/openclaw/session/${sessionId}/files?path=${encodeURIComponent(path)}`);
      setFiles(data.items || []);
    } catch { setFiles([]); }
  };

  const openFileContent = async (path: string) => {
    try {
      const data = await apiFetch(`/openclaw/session/${sessionId}/read?path=${encodeURIComponent(path)}`);
      setOpenFile(data);
      setActiveTab("editor");
    } catch {}
  };


  const run = async () => {
    if (!task.trim() || running) return;
    setRunning(true);
    const taskText = task;
    setTask("");
    setEvents(prev => [...prev, { type: "user", content: taskText, ts: Date.now() }]);
    abortRef.current = new AbortController();

    try {
      const response = await fetch(`/openclaw/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: taskText, model, provider, cwd, session_id: sessionId, stream: true }),
        signal: abortRef.current.signal,
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.event_type === "stream_token") {
              setEvents(prev => {
                const last = prev[prev.length - 1];
                if (last?.type === "stream") return [...prev.slice(0, -1), { ...last, content: last.content + evt.content }];
                return [...prev, { type: "stream", content: evt.content, ts: Date.now() }];
              });
            } else if (evt.event_type === "tool_call") {
              setEvents(prev => [...prev, { type: "tool_call", content: evt.content, ts: Date.now() }]);
            } else if (evt.event_type === "tool_result") {
              setEvents(prev => [...prev, { type: "tool_result", content: evt.content, ts: Date.now() }]);
              // Refresh files after write/edit
              const tool = evt.content?.tool || "";
              if (["write_file","edit_file"].includes(tool)) loadFiles(cwd);
            } else if (evt.event_type === "complete") {
              setEvents(prev => [...prev, { type: "complete", content: evt.content, ts: Date.now() }]);
            } else if (evt.event_type === "error") {
              setEvents(prev => [...prev, { type: "error", content: evt.content, ts: Date.now() }]);
            }
          } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") setEvents(prev => [...prev, { type: "error", content: e.message, ts: Date.now() }]);
    } finally {
      setRunning(false);
    }
  };

  const TOOL_ICONS: Record<string, string> = {
    read_file: "📖", write_file: "✍️", edit_file: "✏️",
    run_command: "⚡", search_files: "🔍", list_directory: "📁",
    git_command: "🔀", glob_files: "🗂️",
  };

  const suggestions = [
    "List all files and explain the project structure",
    "Find all TODO comments and list them",
    "Check git status and show recent commits",
    "Write a README.md for this project",
    "Find any obvious bugs or security issues",
    "Add type hints to all Python functions",
  ];

  const EXT_COLORS: Record<string, string> = {
    py: "text-yellow-400", ts: "text-blue-400", tsx: "text-cyan-400",
    js: "text-yellow-300", rs: "text-orange-400", go: "text-cyan-300",
    json: "text-green-400", md: "text-purple-400", sh: "text-green-300",
    css: "text-pink-400", html: "text-orange-300", sql: "text-blue-300",
    toml: "text-red-300", yaml: "text-yellow-200", yml: "text-yellow-200",
  };

  const fileExt = (name: string) => name.split(".").pop() || "";
  const fileColor = (name: string) => EXT_COLORS[fileExt(name)] || "text-neutral-400";
  const fileIcon = (f: {name: string; is_dir: boolean}) => {
    if (f.is_dir) return "📁";
    const ext = fileExt(f.name);
    const icons: Record<string, string> = {
      py: "🐍", ts: "📘", tsx: "⚛️", js: "🟨", rs: "🦀", go: "🐹",
      json: "📋", md: "📝", sh: "⚡", css: "🎨", html: "🌐",
      sql: "🗃️", dockerfile: "🐳", lock: "🔒",
    };
    return icons[ext] || icons[f.name.toLowerCase()] || "📄";
  };

  const humanSize = (b: number) => b > 1024*1024 ? `${(b/1024/1024).toFixed(1)}M` : b > 1024 ? `${(b/1024).toFixed(0)}K` : `${b}B`;

  return (
    <div className="flex h-full bg-neutral-950 text-sm">
      {/* LEFT — File Explorer */}
      <div className="w-52 flex-shrink-0 border-r border-neutral-800 flex flex-col bg-[#0d0d0d]">
        <div className="px-3 py-2 border-b border-neutral-800 flex items-center gap-2">
          <span className="text-[10px] text-neutral-600 uppercase tracking-wider font-semibold">Explorer</span>
          <button onClick={() => loadFiles(cwd)} className="ml-auto text-neutral-700 hover:text-neutral-400 transition-colors">
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
        {/* CWD breadcrumb */}
        <div className="px-3 py-1.5 border-b border-neutral-900">
          <input value={cwd} onChange={e => setCwd(e.target.value)} onBlur={() => loadFiles(cwd)}
            className="w-full bg-transparent text-[10px] text-neutral-600 focus:outline-none focus:text-neutral-400 font-mono truncate" />
        </div>
        <div className="flex-1 overflow-y-auto">
          {files.length === 0 ? (
            <p className="text-[10px] text-neutral-700 px-3 py-4">No files found</p>
          ) : files.map(f => (
            <button key={f.path} onClick={() => f.is_dir ? (setCwd(f.path), loadFiles(f.path)) : openFileContent(f.path)}
              className="w-full text-left flex items-center gap-1.5 px-3 py-1 hover:bg-neutral-800/60 transition-colors group">
              <span className="text-sm flex-shrink-0">{fileIcon(f)}</span>
              <span className={cx("text-[11px] truncate flex-1", f.is_dir ? "text-neutral-300" : fileColor(f.name))}>
                {f.name}
              </span>
              {!f.is_dir && <span className="text-[9px] text-neutral-700 opacity-0 group-hover:opacity-100">{humanSize(f.size)}</span>}
            </button>
          ))}
        </div>
        {/* Parent dir button */}
        <div className="border-t border-neutral-800 p-2">
          <button onClick={() => { const p = cwd.split("/").slice(0,-1).join("/") || "/"; setCwd(p); loadFiles(p); }}
            className="w-full text-[10px] text-neutral-700 hover:text-neutral-400 transition-colors text-left px-1">
            ↑ parent directory
          </button>
        </div>
      </div>

      {/* RIGHT — Main Panel */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex-shrink-0 flex items-center gap-3 px-4 py-2 border-b border-neutral-800 bg-neutral-950">
          <Terminal className="w-4 h-4 text-cyan-400 flex-shrink-0" />
          <span className="text-xs font-bold text-cyan-400">OpenClaw</span>
          <span className="text-[10px] text-neutral-600 border border-neutral-800 px-1.5 py-0.5 rounded">agentic coding agent</span>
          <div className="ml-auto flex items-center gap-2">
            {/* Tab switcher */}
            {(["agent", "files", "editor"] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className={cx("text-[11px] px-2.5 py-1 rounded transition-colors capitalize",
                  activeTab === tab ? "bg-neutral-800 text-white" : "text-neutral-600 hover:text-neutral-400")}>
                {tab}{tab === "editor" && openFile ? `: ${openFile.path.split("/").pop()}` : ""}
              </button>
            ))}
            <div className="w-px h-4 bg-neutral-800" />
            {/* Provider selector */}
            <select value={provider} onChange={e => { setProvider(e.target.value); setModel(""); }}
              className="bg-neutral-900 border border-neutral-800 text-[11px] text-neutral-400 rounded px-2 py-1 focus:outline-none max-w-[120px]">
              {providers.map(p => (
                <option key={p.id} value={p.id} disabled={!p.configured}>
                  {p.icon} {p.name}{!p.configured ? " (no key)" : ""}
                </option>
              ))}
            </select>
            {/* Model selector */}
            <select value={model} onChange={e => setModel(e.target.value)}
              className="bg-neutral-900 border border-neutral-800 text-[11px] text-neutral-400 rounded px-2 py-1 focus:outline-none max-w-[180px]">
              {models.length > 0
                ? models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)
                : <option value="">Loading models…</option>}
            </select>
            {/* Connection test */}
            <button onClick={testProvider} title="Test provider connection"
              className="text-neutral-700 hover:text-neutral-400 transition-colors">
              {checkResult === null ? <span className="text-xs">⚡</span>
                : checkResult.ok ? <span className="text-[10px] text-green-500">✓{checkResult.latency_ms}ms</span>
                : <span className="text-[10px] text-red-500" title={checkResult.error}>✗</span>}
            </button>
          </div>
        </div>

        {/* Agent tab */}
        {activeTab === "agent" && (
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-1 overflow-y-auto p-4 space-y-2 font-mono">
              {events.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full gap-5">
                  <div className="text-center">
                    <p className="text-5xl mb-3">🦅</p>
                    <p className="text-neutral-300 font-semibold text-sm">OpenClaw ready</p>
                    <p className="text-neutral-600 text-xs mt-1">
                      {providers.find(p => p.id === provider)?.configured
                        ? `${providers.find(p=>p.id===provider)?.icon} ${providers.find(p=>p.id===provider)?.name} · ${model || "select a model"}`
                        : "Add an API key in Settings to get started"}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-2 w-full max-w-xl">
                    {suggestions.map((s, i) => (
                      <button key={i} onClick={() => setTask(s)}
                        className="text-left px-3 py-2 rounded-lg border border-neutral-800 hover:border-cyan-700/50 hover:bg-neutral-900/80 transition-all text-[11px] text-neutral-600 hover:text-neutral-300">
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {events.map((evt, i) => (
                <div key={i} className="min-w-0">
                  {evt.type === "user" && (
                    <div className="flex gap-2 py-2 border-b border-neutral-900 mb-2">
                      <span className="text-cyan-600 flex-shrink-0 text-base">❯</span>
                      <span className="text-neutral-200 text-xs leading-relaxed">{evt.content}</span>
                    </div>
                  )}
                  {evt.type === "tool_call" && (
                    <div className="flex items-center gap-2 text-[11px] py-0.5 text-neutral-700">
                      <span>{TOOL_ICONS[evt.content?.tool] || "🔧"}</span>
                      <span className="text-cyan-700">{evt.content?.tool}</span>
                      <span className="truncate text-neutral-800">
                        {Object.entries(evt.content?.args || {}).slice(0,2).map(([k,v]) =>
                          `${k}=${JSON.stringify(v).slice(0,50)}`
                        ).join("  ")}
                      </span>
                    </div>
                  )}
                  {evt.type === "tool_result" && (
                    <div className="ml-5 mb-1">
                      <div className="bg-[#0a0a0a] border border-neutral-900 rounded px-3 py-2 text-[10px] text-neutral-600 font-mono max-h-24 overflow-y-auto whitespace-pre-wrap">
                        {String(evt.content?.result || "").slice(0, 800)}
                      </div>
                    </div>
                  )}
                  {evt.type === "stream" && (
                    <div className="text-xs text-neutral-300 leading-relaxed whitespace-pre-wrap py-1">
                      {evt.content}
                      {running && i === events.length - 1 && (
                        <span className="inline-block w-1.5 h-3 bg-cyan-400 ml-0.5 animate-pulse align-text-bottom" />
                      )}
                    </div>
                  )}
                  {evt.type === "complete" && (
                    <div className="text-[10px] text-neutral-700 flex items-center gap-1 mt-1 pb-3 border-b border-neutral-900">
                      <span className="text-green-600">✓</span> Done
                    </div>
                  )}
                  {evt.type === "error" && (
                    <div className="bg-red-950/30 border border-red-800/30 rounded px-3 py-2 text-[11px] text-red-400">
                      {evt.content}
                    </div>
                  )}
                </div>
              ))}
              <div ref={eventsEndRef} />
            </div>

            {/* Input */}
            <div className="flex-shrink-0 border-t border-neutral-800 p-3 bg-[#0d0d0d]">
              <div className="flex gap-2 items-end">
                <textarea value={task} onChange={e => setTask(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } }}
                  placeholder="Describe a coding task… (Enter to run, Shift+Enter for newline)"
                  rows={2}
                  className="flex-1 bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-xs text-neutral-300 placeholder-neutral-700 focus:outline-none focus:border-cyan-700/50 resize-none font-mono"
                />
                {running ? (
                  <button onClick={() => abortRef.current?.abort()}
                    className="px-3 py-2 rounded-lg bg-red-900/30 border border-red-700/40 text-red-400 text-xs hover:bg-red-900/50 transition-colors flex-shrink-0">
                    ✕ Stop
                  </button>
                ) : (
                  <button onClick={run} disabled={!task.trim() || !model || running}
                    className="px-3 py-2 rounded-lg bg-cyan-900/30 border border-cyan-700/40 text-cyan-400 text-xs hover:bg-cyan-900/50 transition-colors disabled:opacity-30 flex-shrink-0">
                    Run ⏎
                  </button>
                )}
              </div>
              {events.length > 0 && (
                <div className="flex gap-3 mt-2">
                  <button onClick={() => setEvents([])} className="text-[10px] text-neutral-700 hover:text-neutral-500 transition-colors">
                    Clear history
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Files tab — shows files as cards */}
        {activeTab === "files" && (
          <div className="flex-1 overflow-y-auto p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs text-neutral-500 font-mono">{cwd}</span>
              <button onClick={() => loadFiles(cwd)} className="text-[10px] text-neutral-700 hover:text-neutral-400 transition-colors">
                ↺ refresh
              </button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {files.map(f => (
                <button key={f.path} onClick={() => f.is_dir ? (setCwd(f.path)) : openFileContent(f.path)}
                  className="text-left p-3 rounded-lg border border-neutral-800 hover:border-neutral-700 hover:bg-neutral-900/50 transition-all">
                  <div className="text-xl mb-1">{fileIcon(f)}</div>
                  <div className={cx("text-[11px] font-mono truncate", f.is_dir ? "text-neutral-300" : fileColor(f.name))}>
                    {f.name}
                  </div>
                  {!f.is_dir && <div className="text-[9px] text-neutral-700 mt-0.5">{humanSize(f.size)}</div>}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Editor tab — read-only file viewer */}
        {activeTab === "editor" && (
          <div className="flex-1 flex flex-col min-h-0">
            {openFile ? (
              <>
                <div className="flex-shrink-0 flex items-center gap-3 px-4 py-2 border-b border-neutral-800 bg-[#0d0d0d]">
                  <span className="text-[10px] text-neutral-600 font-mono truncate">{openFile.path}</span>
                  <span className="ml-auto text-[10px] text-neutral-700">{openFile.lines} lines</span>
                  <button onClick={() => { setTask(`Edit ${openFile.path}: `); setActiveTab("agent"); }}
                    className="text-[10px] text-cyan-700 hover:text-cyan-500 transition-colors">
                    Ask agent to edit →
                  </button>
                </div>
                <div className="flex-1 overflow-auto p-0">
                  <pre className="text-[11px] font-mono text-neutral-400 leading-5 p-4 min-w-max">
                    {openFile.content.split("\n").map((line, i) => (
                      <div key={i} className="flex">
                        <span className="w-10 flex-shrink-0 text-right pr-3 text-neutral-700 select-none">{i + 1}</span>
                        <span className="flex-1">{line}</span>
                      </div>
                    ))}
                  </pre>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-neutral-700 text-sm">
                Click a file in the explorer to view it
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}




// ══════════════════════════════════════════════════════════════════════════════
// SETTINGS PAGE — Production-grade setup wizard
// ══════════════════════════════════════════════════════════════════════════════

type ProviderCard = {
  id: string; name: string; icon: string; configured: boolean;
  masked: string; key_length: number; description: string;
  free_tier: boolean; get_key_url: string; docs: string;
  env_var: string; auth_method?: string; oauth_email?: string; oauth_name?: string;
};

function SetupWizard({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(0);
  const steps = ["Welcome", "Ollama", "Cloud APIs", "Test & Launch"];

  return (
    <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0d0d0f] border border-white/10 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden">
        {/* Progress bar */}
        <div className="h-0.5 bg-white/5">
          <div className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-500"
               style={{ width: `${((step + 1) / steps.length) * 100}%` }} />
        </div>
        {/* Step indicators */}
        <div className="flex items-center gap-0 px-8 pt-6 pb-2">
          {steps.map((s, i) => (
            <div key={s} className="flex items-center gap-0 flex-1 last:flex-none">
              <div className={cx("flex items-center gap-2", i < steps.length - 1 ? "flex-1" : "")}>
                <div className={cx(
                  "w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0 transition-colors",
                  i < step ? "bg-cyan-500 text-black" :
                  i === step ? "bg-white/10 text-white border border-white/20" :
                  "bg-white/5 text-white/30"
                )}>
                  {i < step ? "✓" : i + 1}
                </div>
                <span className={cx("text-xs transition-colors whitespace-nowrap",
                  i === step ? "text-white" : i < step ? "text-cyan-400" : "text-white/30")}>
                  {s}
                </span>
                {i < steps.length - 1 && (
                  <div className={cx("h-px flex-1 mx-3 transition-colors",
                    i < step ? "bg-cyan-500/50" : "bg-white/10")} />
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Step content */}
        <div className="px-8 py-6">
          {step === 0 && <WizardWelcome onNext={() => setStep(1)} />}
          {step === 1 && <WizardOllama onNext={() => setStep(2)} onBack={() => setStep(0)} />}
          {step === 2 && <WizardCloudAPIs onNext={() => setStep(3)} onBack={() => setStep(1)} />}
          {step === 3 && <WizardLaunch onDone={onDone} onBack={() => setStep(2)} />}
        </div>
      </div>
    </div>
  );
}

function WizardWelcome({ onNext }: { onNext: () => void }) {
  return (
    <div className="text-center py-4">
      <div className="text-5xl mb-4">🚀</div>
      <h2 className="text-2xl font-bold text-white mb-2">Welcome to AgentFlow</h2>
      <p className="text-neutral-400 text-sm max-w-md mx-auto mb-8">
        Let's connect your AI providers. This takes about 2 minutes. You can always change these in Settings later.
      </p>
      <div className="grid grid-cols-3 gap-3 mb-8 text-left">
        {[
          { icon: "🦙", label: "Ollama", desc: "Local models, zero cost" },
          { icon: "☁️", label: "Cloud APIs", desc: "Groq, OpenAI, Gemini, more" },
          { icon: "⚡", label: "One config", desc: "All agents share your keys" },
        ].map(item => (
          <div key={item.label} className="bg-white/5 rounded-xl p-4 border border-white/10">
            <div className="text-2xl mb-2">{item.icon}</div>
            <div className="text-sm font-medium text-white">{item.label}</div>
            <div className="text-xs text-neutral-500 mt-0.5">{item.desc}</div>
          </div>
        ))}
      </div>
      <button onClick={onNext}
        className="px-8 py-3 bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-semibold rounded-xl hover:opacity-90 transition-opacity">
        Get Started →
      </button>
    </div>
  );
}

function WizardOllama({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const [url, setUrl] = useState("http://localhost:11434");
  const [key, setKey] = useState("");
  const [status, setStatus] = useState<any>(null);
  const [testing, setTesting] = useState(false);

  const test = async () => {
    setTesting(true); setStatus(null);
    try {
      const r = await apiFetch(`/settings/ollama?url=${encodeURIComponent(url)}&api_key=${encodeURIComponent(key)}`, { method: "POST" });
      setStatus(r);
    } catch { setStatus({ available: false, error: "Connection failed" }); }
    setTesting(false);
  };

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <span className="text-3xl">🦙</span>
        <div>
          <h3 className="text-lg font-bold text-white">Ollama — Local Models</h3>
          <p className="text-sm text-neutral-400">Free, private, runs on your machine or a remote server</p>
        </div>
      </div>
      <div className="space-y-3 mb-6">
        <div>
          <label className="text-xs text-neutral-500 mb-1 block">Ollama URL</label>
          <input value={url} onChange={e => setUrl(e.target.value)}
            placeholder="http://localhost:11434"
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-cyan-500/50 font-mono" />
        </div>
        <div>
          <label className="text-xs text-neutral-500 mb-1 block">API Key <span className="text-white/20">(optional — only for secured endpoints)</span></label>
          <input type="password" value={key} onChange={e => setKey(e.target.value)}
            placeholder="Bearer token if your Ollama is behind auth"
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-cyan-500/50 font-mono" />
        </div>
        <button onClick={test} disabled={testing}
          className="w-full py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-sm text-white transition-colors disabled:opacity-50">
          {testing ? "Testing…" : "Test Connection"}
        </button>
        {status && (
          <div className={cx("rounded-xl px-4 py-3 text-sm flex items-center gap-3",
            status.available ? "bg-green-500/10 border border-green-500/20 text-green-300"
                             : "bg-amber-500/10 border border-amber-500/20 text-amber-300")}>
            <span>{status.available ? "✓" : "○"}</span>
            <span>{status.available
              ? `Connected — ${status.model_count} models available`
              : `Not reachable — ${status.error || "Ollama may not be running"}`}
            </span>
          </div>
        )}
      </div>
      <div className="flex gap-3">
        <button onClick={onBack} className="px-4 py-2.5 bg-white/5 text-white/50 rounded-xl text-sm hover:text-white transition-colors">← Back</button>
        <button onClick={onNext} className="flex-1 py-2.5 bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-semibold rounded-xl hover:opacity-90 transition-opacity">
          {status?.available ? "Continue →" : "Skip for now →"}
        </button>
      </div>
    </div>
  );
}

function WizardCloudAPIs({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const [saved, setSaved] = useState<Record<string, boolean>>({});
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<string | null>(null);

  const providers = [
    { id: "groq", icon: "⚡", name: "Groq", tag: "FREE · Fastest", color: "text-yellow-400", desc: "Llama 3.3 70B at 500 tok/s", url: "https://console.groq.com/keys", placeholder: "gsk_..." },
    { id: "gemini", icon: "🔵", name: "Google Gemini", tag: "FREE tier", color: "text-blue-400", desc: "Gemini 2.0 Flash", url: "https://aistudio.google.com/app/apikey", placeholder: "AIza..." },
    { id: "openai", icon: "🟢", name: "OpenAI", tag: "Pay-per-use", color: "text-green-400", desc: "GPT-4o, o4-mini", url: "https://platform.openai.com/api-keys", placeholder: "sk-..." },
    { id: "anthropic", icon: "🟠", name: "Anthropic", tag: "Pay-per-use", color: "text-orange-400", desc: "Claude Sonnet, Haiku", url: "https://console.anthropic.com/settings/keys", placeholder: "sk-ant-..." },
    { id: "together", icon: "🟣", name: "Together AI", tag: "FREE $25 credit", color: "text-violet-400", desc: "100+ open models", url: "https://api.together.xyz/settings/api-keys", placeholder: "..." },
    { id: "openrouter", icon: "🔗", name: "OpenRouter", tag: "FREE tier", color: "text-slate-400", desc: "300+ models, one key", url: "https://openrouter.ai/settings/keys", placeholder: "sk-or-..." },
  ];

  const save = async (id: string) => {
    const val = inputs[id]?.trim();
    if (!val) return;
    setSaving(id);
    try {
      await apiFetch("/settings/keys", { method: "POST", body: JSON.stringify({ keys: { [id]: val } }) });
      setSaved(p => ({ ...p, [id]: true }));
      setInputs(p => ({ ...p, [id]: "" }));
    } catch { }
    setSaving(null);
  };

  return (
    <div>
      <div className="mb-5">
        <h3 className="text-lg font-bold text-white">Cloud API Keys</h3>
        <p className="text-sm text-neutral-400 mt-0.5">Add at least one to start. All are optional — add more any time in Settings.</p>
      </div>
      <div className="space-y-2 mb-6 max-h-72 overflow-y-auto pr-1">
        {providers.map(p => (
          <div key={p.id} className={cx(
            "rounded-xl border p-3 transition-colors",
            saved[p.id] ? "bg-green-500/5 border-green-500/20" : "bg-white/3 border-white/8"
          )}>
            <div className="flex items-center gap-3">
              <span className="text-xl">{p.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-white">{p.name}</span>
                  <span className={cx("text-[10px] font-mono", p.color)}>{p.tag}</span>
                  {saved[p.id] && <span className="text-[10px] text-green-400">✓ saved</span>}
                </div>
                <span className="text-xs text-neutral-500">{p.desc}</span>
              </div>
              <a href={p.url} target="_blank" rel="noreferrer"
                className="text-[10px] text-cyan-600 hover:text-cyan-400 whitespace-nowrap">
                Get key →
              </a>
            </div>
            {!saved[p.id] && (
              <div className="flex gap-2 mt-2">
                <input type="password"
                  value={inputs[p.id] || ""}
                  onChange={e => setInputs(prev => ({ ...prev, [p.id]: e.target.value }))}
                  onKeyDown={e => e.key === "Enter" && save(p.id)}
                  placeholder={p.placeholder}
                  className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-white/20 focus:outline-none focus:border-cyan-500/50 font-mono" />
                <button onClick={() => save(p.id)}
                  disabled={!inputs[p.id]?.trim() || saving === p.id}
                  className="px-3 py-1.5 bg-white/10 hover:bg-white/15 disabled:opacity-30 text-white text-xs rounded-lg transition-colors">
                  {saving === p.id ? "…" : "Save"}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="flex gap-3">
        <button onClick={onBack} className="px-4 py-2.5 bg-white/5 text-white/50 rounded-xl text-sm hover:text-white transition-colors">← Back</button>
        <button onClick={onNext} className="flex-1 py-2.5 bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-semibold rounded-xl hover:opacity-90 transition-opacity">
          {Object.keys(saved).length > 0 ? `Continue with ${Object.keys(saved).length} provider${Object.keys(saved).length > 1 ? "s" : ""} →` : "Skip for now →"}
        </button>
      </div>
    </div>
  );
}

function WizardLaunch({ onDone, onBack }: { onDone: () => void; onBack: () => void }) {
  const [results, setResults] = useState<Record<string, any>>({});
  const [testing, setTesting] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => { runTests(); }, []);

  const runTests = async () => {
    setTesting(true);
    try {
      const r = await apiFetch("/cloud/providers/check", { method: "POST" });
      setResults(r);
    } catch { }
    setTesting(false);
    setDone(true);
  };

  const passed = Object.values(results).filter((r: any) => r.configured && r.healthy).length;
  const total = Object.values(results).filter((r: any) => r.configured).length;

  return (
    <div className="text-center">
      <div className="text-5xl mb-4">{done ? (passed > 0 ? "🎉" : "🔧") : "⏳"}</div>
      <h3 className="text-lg font-bold text-white mb-1">
        {testing ? "Testing connections…" : done ? (passed > 0 ? "You're all set!" : "Almost there") : ""}
      </h3>
      <p className="text-sm text-neutral-400 mb-6">
        {done && passed > 0 && `${passed} of ${total} providers connected and working.`}
        {done && passed === 0 && "No providers configured yet — you can add keys in Settings anytime."}
      </p>

      {/* Live test results */}
      <div className="space-y-2 mb-6 text-left">
        {Object.entries(results).filter(([, r]: [string, any]) => r.configured).map(([pid, r]: [string, any]) => (
          <div key={pid} className={cx(
            "flex items-center gap-3 rounded-xl px-4 py-2.5 border text-sm",
            r.healthy ? "bg-green-500/5 border-green-500/20" : "bg-red-500/5 border-red-500/20"
          )}>
            <span className={r.healthy ? "text-green-400" : "text-red-400"}>
              {r.healthy ? "✓" : "✗"}
            </span>
            <span className="text-white capitalize flex-1">{pid}</span>
            {r.healthy && <span className="text-xs text-neutral-500">{r.latency_ms}ms</span>}
            {!r.healthy && <span className="text-xs text-red-400/70">{r.error?.slice(0, 40)}</span>}
          </div>
        ))}
        {testing && (
          <div className="flex items-center gap-3 rounded-xl px-4 py-2.5 border border-white/10 text-sm">
            <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-white/50">Testing connections…</span>
          </div>
        )}
        {done && total === 0 && (
          <div className="text-center py-4 text-neutral-500 text-sm">
            No API keys added yet — that's fine! Add them in Settings when ready.
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <button onClick={onBack} className="px-4 py-2.5 bg-white/5 text-white/50 rounded-xl text-sm hover:text-white transition-colors">← Back</button>
        <button onClick={onDone}
          className="flex-1 py-2.5 bg-gradient-to-r from-cyan-500 to-violet-500 text-white font-semibold rounded-xl hover:opacity-90 transition-opacity">
          Open AgentFlow →
        </button>
      </div>
    </div>
  );
}

// ── Provider connection card component ────────────────────────────────────────
function ProviderConnectionCard({
  info, pid, editVal, isTesting, testResult,
  onEdit, onSave, onTest, onRemove, saving,
  googleStatus, googleSetupStep, setGoogleSetupStep,
  googleClientId, setGoogleClientId,
  googleClientSecret, setGoogleClientSecret,
  saveGoogleCreds, startGoogleOAuth, revokeGoogle, testGoogle,
  openAIStatus, openAIGuide, openAIKey, setOpenAIKey, connectOpenAI,
}: any) {

  if (pid === "gemini") {
    const isOAuth = info.auth_method === "oauth";
    return (
      <div className={cx("group rounded-2xl border p-5 transition-all duration-300",
        info.configured
          ? "bg-gradient-to-br from-blue-950/30 to-cyan-950/20 border-blue-500/20"
          : "bg-white/[0.02] border-white/8 hover:border-white/15")}>

        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-xl flex-shrink-0">
            🔵
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-white">Google Gemini</span>
              <span className="text-[10px] bg-green-500/15 text-green-400 border border-green-500/25 px-2 py-0.5 rounded-full">FREE</span>
              {info.configured && (
                <span className="text-[10px] bg-blue-500/15 text-blue-400 border border-blue-500/25 px-2 py-0.5 rounded-full">
                  {isOAuth ? "✓ OAuth" : "✓ API key"}
                </span>
              )}
            </div>
            <p className="text-xs text-neutral-500 mt-0.5">Gemini 2.0 Flash, 1.5 Pro — sign in with Google or use an API key</p>
          </div>
          <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer"
            className="text-[10px] text-neutral-600 hover:text-cyan-500 transition-colors whitespace-nowrap opacity-0 group-hover:opacity-100">
            AI Studio →
          </a>
        </div>

        {/* Connected via OAuth */}
        {isOAuth && googleStatus?.connected && (
          <div className="mb-4 flex items-center justify-between bg-green-500/8 border border-green-500/20 rounded-xl px-4 py-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center text-sm font-bold text-black">
                {(googleStatus.user_name || "G")[0].toUpperCase()}
              </div>
              <div>
                <div className="text-sm font-medium text-white">{googleStatus.user_name}</div>
                <div className="text-xs text-neutral-500">{googleStatus.user_email}</div>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={testGoogle}
                className="text-xs px-3 py-1.5 bg-white/8 hover:bg-white/12 text-neutral-300 rounded-lg transition-colors">
                Test ⚡
              </button>
              <button onClick={revokeGoogle}
                className="text-xs px-3 py-1.5 text-red-500/70 hover:text-red-400 transition-colors">
                Sign out
              </button>
            </div>
          </div>
        )}

        {/* Setup options */}
        {!info.configured && (
          <div className="space-y-3">
            {/* OAuth option */}
            <div className="bg-white/4 rounded-xl border border-white/8 p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="text-sm font-medium text-white">Sign in with Google</div>
                  <div className="text-xs text-neutral-500">OAuth2 — no API key, uses your Google account</div>
                </div>
                <span className="text-[10px] text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-full">Recommended</span>
              </div>

              {googleSetupStep === "idle" && (
                <button onClick={() => setGoogleSetupStep("creds")}
                  className="w-full flex items-center justify-center gap-3 py-2.5 bg-white text-neutral-900 text-sm font-semibold rounded-xl hover:bg-neutral-100 transition-colors">
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  Set up Google Sign-in
                </button>
              )}

              {googleSetupStep === "creds" && (
                <div className="space-y-2">
                  <div className="bg-blue-500/8 border border-blue-500/15 rounded-lg p-3 text-xs text-blue-300">
                    <span className="font-medium">One-time setup:</span> Go to{" "}
                    <a href="https://console.cloud.google.com/apis/credentials" target="_blank" className="underline hover:text-blue-200">
                      Google Cloud Console
                    </a>{" "}
                    → Create OAuth 2.0 Client → Web application. Set redirect URI to{" "}
                    <code className="bg-blue-900/40 px-1 rounded text-blue-200">{window.location.origin}/auth/google/callback</code>
                  </div>
                  <input value={googleClientId} onChange={e => setGoogleClientId(e.target.value)}
                    placeholder="Client ID  (…googleusercontent.com)"
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-white/25 focus:outline-none focus:border-blue-500/50 font-mono" />
                  <input type="password" value={googleClientSecret} onChange={e => setGoogleClientSecret(e.target.value)}
                    placeholder="Client Secret"
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-white/25 focus:outline-none focus:border-blue-500/50 font-mono" />
                  <div className="flex gap-2">
                    <button onClick={saveGoogleCreds} disabled={!googleClientId || !googleClientSecret}
                      className="flex-1 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white text-xs font-semibold rounded-lg transition-colors">
                      Save & Continue
                    </button>
                    <button onClick={() => setGoogleSetupStep("idle")}
                      className="px-3 py-2 text-white/40 hover:text-white text-xs transition-colors">Cancel</button>
                  </div>
                </div>
              )}

              {googleSetupStep === "auth" && (
                <button onClick={startGoogleOAuth}
                  className="w-full flex items-center justify-center gap-3 py-2.5 bg-white text-neutral-900 text-sm font-semibold rounded-xl hover:bg-neutral-100 transition-colors">
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  Sign in with Google
                </button>
              )}
            </div>

            {/* API key option */}
            <div className="bg-white/[0.02] rounded-xl border border-white/8 p-4">
              <div className="text-sm font-medium text-white mb-1">Or use an API key</div>
              <div className="text-xs text-neutral-500 mb-3">
                Get a free key from <a href="https://aistudio.google.com/app/apikey" target="_blank" className="text-cyan-600 hover:text-cyan-400">AI Studio</a>
              </div>
              <div className="flex gap-2">
                <input type="password" value={editVal}
                  onChange={e => onEdit(e.target.value)}
                  onKeyDown={(e: any) => e.key === "Enter" && onSave()}
                  placeholder="AIzaSy..."
                  className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-white/25 focus:outline-none focus:border-cyan-500/50 font-mono" />
                <button onClick={onSave} disabled={!editVal}
                  className="px-3 py-2 bg-white/10 hover:bg-white/15 disabled:opacity-30 text-white text-xs rounded-lg transition-colors">
                  Save
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Configured via API key */}
        {info.configured && !isOAuth && (
          <div className="flex items-center gap-3">
            <code className="text-xs text-neutral-400 bg-white/5 px-3 py-1.5 rounded-lg font-mono flex-1">{info.masked}</code>
            {testResult && (
              <span className={cx("text-xs px-2 py-1 rounded-lg", testResult.success ? "text-green-400 bg-green-500/10" : "text-red-400 bg-red-500/10")}>
                {testResult.success ? `✓ ${testResult.latency_ms}ms` : "✗ failed"}
              </span>
            )}
            <button onClick={onTest} disabled={isTesting}
              className="text-xs px-3 py-1.5 bg-white/8 hover:bg-white/12 text-white rounded-lg transition-colors">
              {isTesting ? "…" : "Test"}
            </button>
            <button onClick={onRemove} className="text-xs text-red-500/60 hover:text-red-400 transition-colors">Remove</button>
          </div>
        )}
      </div>
    );
  }

  if (pid === "openai") {
    return (
      <div className={cx("group rounded-2xl border p-5 transition-all duration-300",
        openAIStatus?.valid
          ? "bg-gradient-to-br from-green-950/30 to-emerald-950/20 border-green-500/20"
          : "bg-white/[0.02] border-white/8 hover:border-white/15")}>

        <div className="flex items-start gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center justify-center text-xl flex-shrink-0">🟢</div>
          <div className="flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-white">OpenAI</span>
              {openAIStatus?.valid && <span className="text-[10px] bg-green-500/15 text-green-400 border border-green-500/25 px-2 py-0.5 rounded-full">✓ connected</span>}
              {openAIStatus?.tier && <span className="text-[10px] text-neutral-500 border border-white/10 px-2 py-0.5 rounded-full">{openAIStatus.tier}</span>}
            </div>
            <p className="text-xs text-neutral-500 mt-0.5">GPT-4o, o4-mini — API billing is separate from ChatGPT Plus</p>
          </div>
        </div>

        {!openAIStatus?.valid && (
          <div className="mb-4 flex gap-3 bg-amber-500/8 border border-amber-500/20 rounded-xl px-4 py-3">
            <span className="text-amber-400 text-lg">⚠</span>
            <div>
              <div className="text-sm font-medium text-amber-300">ChatGPT Plus ≠ API Access</div>
              <div className="text-xs text-neutral-500 mt-0.5 leading-relaxed">
                Your $20/mo subscription is for chat.openai.com only. The API has separate billing — same account, minimum $5 top-up, pay per use. GPT-4o Mini costs ~$0.15/1M tokens.
              </div>
            </div>
          </div>
        )}

        {openAIStatus?.valid && (
          <div className="mb-4 flex items-center gap-3 bg-green-500/8 border border-green-500/20 rounded-xl px-4 py-2.5">
            <div className="w-2 h-2 rounded-full bg-green-400" />
            <code className="text-xs text-neutral-400 font-mono flex-1">{openAIStatus.masked_key}</code>
            <span className="text-xs text-neutral-600">{openAIStatus.model_count} models · {openAIStatus.latency_ms}ms</span>
            <button onClick={() => onRemove()} className="text-xs text-red-500/60 hover:text-red-400 transition-colors ml-2">Remove</button>
          </div>
        )}

        {!openAIStatus?.valid && openAIGuide && (
          <div className="mb-4 grid grid-cols-2 gap-2">
            {openAIGuide.steps.map((s: any) => (
              <div key={s.step} className="flex gap-2.5 text-xs">
                <span className="w-5 h-5 rounded-full bg-white/8 border border-white/12 flex items-center justify-center text-[10px] text-white/50 flex-shrink-0">{s.step}</span>
                <div>
                  <span className="text-white/70 font-medium">{s.title}</span>
                  {s.url && <a href={s.url} target="_blank" rel="noreferrer" className="ml-1 text-cyan-600 hover:text-cyan-400">↗</a>}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <input type="password" value={openAIKey}
            onChange={e => setOpenAIKey(e.target.value)}
            onKeyDown={(e: any) => e.key === "Enter" && connectOpenAI()}
            placeholder="sk-proj-... or sk-..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/25 focus:outline-none focus:border-green-500/50 font-mono" />
          <button onClick={connectOpenAI} disabled={!openAIKey.trim()}
            className="px-4 py-2.5 bg-green-700 hover:bg-green-600 disabled:opacity-30 text-white text-sm font-semibold rounded-xl transition-colors whitespace-nowrap">
            Connect
          </button>
        </div>

        {openAIGuide?.cost_estimate && (
          <div className="mt-3 flex gap-2 flex-wrap">
            {Object.entries(openAIGuide.cost_estimate).map(([m, cost]: [string, any]) => (
              <div key={m} className="bg-white/4 rounded-lg px-2 py-1 text-[10px]">
                <span className="text-white/50">{m.replace(/_/g," ")}</span>
                <span className="text-white/30 ml-1">{cost}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Generic provider card
  const COLORS: Record<string, string> = {
    groq: "yellow", together: "violet", mistral: "red", anthropic: "orange", openrouter: "slate"
  };
  const col = COLORS[pid] || "neutral";
  const borderClass = info.configured ? `border-${col}-500/20` : "border-white/8";

  return (
    <div className={cx("group rounded-2xl border p-5 transition-all duration-300 bg-white/[0.02] hover:border-white/15", borderClass)}>
      <div className="flex items-start gap-3">
        <div className={cx(`w-10 h-10 rounded-xl bg-${col}-500/10 border border-${col}-500/20 flex items-center justify-center text-xl flex-shrink-0`)}>
          {info.icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-0.5">
            <span className="font-semibold text-white">{info.name}</span>
            {info.free_tier && <span className="text-[10px] bg-green-500/12 text-green-400 border border-green-500/20 px-2 py-0.5 rounded-full">FREE</span>}
            {info.configured && <span className="text-[10px] bg-cyan-500/12 text-cyan-400 border border-cyan-500/20 px-2 py-0.5 rounded-full">✓ active</span>}
          </div>
          <p className="text-xs text-neutral-500">{info.description}</p>
        </div>
        <a href={info.get_key_url} target="_blank" rel="noreferrer"
          className="text-[10px] text-neutral-600 hover:text-cyan-500 transition-colors whitespace-nowrap opacity-0 group-hover:opacity-100">
          Get key →
        </a>
      </div>

      {info.configured && (
        <div className="mt-3 flex items-center gap-2">
          <code className="text-xs text-neutral-500 bg-white/5 px-3 py-1.5 rounded-lg font-mono flex-1">{info.masked}</code>
          {testResult && (
            <span className={cx("text-xs px-2 py-1 rounded-lg", testResult.success ? "text-green-400 bg-green-500/10" : "text-red-400 bg-red-500/10")}>
              {testResult.success ? `✓ ${testResult.latency_ms}ms` : "✗ err"}
            </span>
          )}
          <button onClick={onTest} disabled={isTesting}
            className="text-xs px-3 py-1.5 bg-white/8 hover:bg-white/12 text-white rounded-lg transition-colors">
            {isTesting ? "…" : "Test"}
          </button>
          <button onClick={onRemove} className="text-xs text-red-500/50 hover:text-red-400 transition-colors">✕</button>
        </div>
      )}

      <div className={cx("flex gap-2 transition-all duration-200", info.configured ? "mt-2" : "mt-3")}>
        <input type="password" value={editVal}
          onChange={e => onEdit(e.target.value)}
          onKeyDown={(e: any) => e.key === "Enter" && onSave()}
          placeholder={info.configured ? "Replace key…" : `Paste your ${info.name} API key…`}
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/25 focus:outline-none focus:border-cyan-500/50 font-mono" />
        <button onClick={onSave} disabled={!editVal || saving}
          className="px-4 py-2.5 bg-white/10 hover:bg-white/15 disabled:opacity-30 text-white text-sm rounded-xl transition-colors">
          {saving ? "…" : "Save"}
        </button>
      </div>

      <div className="mt-2 flex items-center gap-3">
        <a href={info.get_key_url} target="_blank" rel="noreferrer" className="text-[11px] text-cyan-700 hover:text-cyan-500 transition-colors">Get key →</a>
        <a href={info.docs} target="_blank" rel="noreferrer" className="text-[11px] text-neutral-700 hover:text-neutral-500 transition-colors">Docs</a>
        <span className="text-[10px] text-neutral-800 font-mono ml-auto">{info.env_var}</span>
      </div>
    </div>
  );
}

// ── Main SettingsPage ─────────────────────────────────────────────────────────
function SettingsPage({ stats }: { stats: any }) {
  const [keys, setKeys] = useState<Record<string, any>>({});
  const [editing, setEditing] = useState<Record<string, string>>({});
  const [testing, setTesting] = useState<Record<string, boolean>>({});
  const [testResults, setTestResults] = useState<Record<string, any>>({});
  const [saving, setSaving] = useState(false);
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [ollamaStatus, setOllamaStatus] = useState<any>(null);
  const [showWizard, setShowWizard] = useState(false);
  // Auth provider states
  const [googleStatus, setGoogleStatus] = useState<any>(null);
  const [googleClientId, setGoogleClientId] = useState("");
  const [googleClientSecret, setGoogleClientSecret] = useState("");
  const [googleSetupStep, setGoogleSetupStep] = useState<"idle"|"creds"|"auth">("idle");
  const [openAIGuide, setOpenAIGuide] = useState<any>(null);
  const [openAIStatus, setOpenAIStatus] = useState<any>(null);
  const [openAIKey, setOpenAIKey] = useState("");
  const [oauthWindow, setOauthWindow] = useState<Window|null>(null);

  const load = async () => {
    try {
      const data = await apiFetch("/settings/keys");
      setKeys(data);
      const edits: Record<string, string> = {};
      Object.keys(data).forEach(k => { edits[k] = ""; });
      setEditing(edits);
    } catch {}
    try {
      const o = await apiFetch("/settings/ollama");
      setOllamaUrl(o.url || "http://localhost:11434");
      setOllamaStatus(o);
    } catch {}
    try {
      const gs = await apiFetch("/auth/google/status");
      setGoogleStatus(gs);
    } catch {}
    try {
      const os2 = await apiFetch("/auth/openai/status");
      setOpenAIStatus(os2);
    } catch {}
    try {
      const guide = await apiFetch("/auth/openai/guide");
      setOpenAIGuide(guide);
    } catch {}
  };

  useEffect(() => { load(); }, []);

  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === "google_auth_complete") {
        load();
        if (oauthWindow) oauthWindow.close();
        toast.success(`Connected as ${e.data.name} (${e.data.email})`);
      }
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [oauthWindow]);

  const startGoogleOAuth = async () => {
    try {
      const r = await apiFetch("/auth/google/start");
      const w = window.open(r.url, "google_auth", "width=600,height=700,scrollbars=yes,resizable=yes");
      setOauthWindow(w);
    } catch (e: any) { toast.error(e.message || "Failed to start Google auth"); }
  };

  const revokeGoogle = async () => {
    await apiFetch("/auth/google/revoke", { method: "POST" });
    await load();
    toast.success("Google account disconnected");
  };

  const testGoogle = async () => {
    const r = await apiFetch("/auth/google/test");
    if (r.ok) toast.success(`Gemini works! ${r.latency_ms}ms`);
    else toast.error(`Gemini test failed: ${r.error}`);
  };

  const saveGoogleCreds = async () => {
    if (!googleClientId || !googleClientSecret) return;
    await apiFetch("/auth/google/credentials", {
      method: "POST",
      body: JSON.stringify({ client_id: googleClientId, client_secret: googleClientSecret }),
    });
    await load();
    setGoogleSetupStep("auth");
    toast.success("Credentials saved — now sign in with Google");
  };

  const connectOpenAI = async () => {
    if (!openAIKey.trim()) return;
    try {
      const r = await apiFetch("/auth/openai/setup", {
        method: "POST", body: JSON.stringify({ api_key: openAIKey.trim() }),
      });
      setOpenAIKey(""); await load(); toast.success(r.message);
    } catch (e: any) { toast.error(e.message || "Invalid key"); }
  };

  const saveKey = async (pid: string) => {
    const val = editing[pid]?.trim();
    if (!val) return;
    setSaving(true);
    try {
      await apiFetch("/settings/keys", { method: "POST", body: JSON.stringify({ keys: { [pid]: val } }) });
      setEditing(prev => ({ ...prev, [pid]: "" }));
      await load();
      toast.success(`${keys[pid]?.name || pid} key saved!`);
    } catch (e: any) { toast.error(e.message || "Failed to save"); }
    finally { setSaving(false); }
  };

  const removeKey = async (pid: string) => {
    try {
      await apiFetch(`/settings/keys/${pid}`, { method: "DELETE" });
      await load(); toast.success("Key removed");
    } catch {}
  };

  const testKey = async (pid: string) => {
    setTesting(prev => ({ ...prev, [pid]: true }));
    setTestResults(prev => ({ ...prev, [pid]: null }));
    try {
      const r = await apiFetch(`/settings/keys/${pid}/test`, { method: "POST" });
      setTestResults(prev => ({ ...prev, [pid]: r }));
    } catch (e: any) {
      setTestResults(prev => ({ ...prev, [pid]: { success: false, error: e.message } }));
    } finally { setTesting(prev => ({ ...prev, [pid]: false })); }
  };

  const providerList = Object.values(keys) as ProviderCard[];
  const configured = providerList.filter(p => p.configured);
  const ollamaOk = ollamaStatus?.available;
  const totalActive = configured.length + (ollamaOk ? 1 : 0);

  return (
    <div className="flex-1 overflow-y-auto bg-[#080809]">
      {showWizard && <SetupWizard onDone={() => { setShowWizard(false); load(); }} />}

      <div className="max-w-2xl mx-auto px-6 py-8 space-y-6">

        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">Connections</h1>
            <p className="text-neutral-500 text-sm mt-0.5">
              {totalActive > 0
                ? `${totalActive} provider${totalActive > 1 ? "s" : ""} active · Changes apply instantly`
                : "No providers connected yet"}
            </p>
          </div>
          <button onClick={() => setShowWizard(true)}
            className="flex items-center gap-2 px-4 py-2 bg-white/8 hover:bg-white/12 border border-white/10 rounded-xl text-sm text-white transition-colors">
            <span>🚀</span> Setup Wizard
          </button>
        </div>

        {/* System status bar */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "Ollama", value: ollamaOk ? `${ollamaStatus?.model_count || 0} models` : "Offline",
              sub: ollamaOk ? `${ollamaStatus?.latency_ms}ms` : "Not connected",
              ok: ollamaOk, icon: "🦙" },
            { label: "Cloud", value: `${configured.length} active`,
              sub: configured.length > 0 ? configured.map((p: any) => p.icon).join(" ") : "No keys set",
              ok: configured.length > 0, icon: "☁️" },
            { label: "Mode", value: stats?.mode || "demo",
              sub: stats?.mode === "hybrid" ? "Local + Cloud" : stats?.mode === "local" ? "Ollama only" : "Add a provider",
              ok: stats?.mode !== "demo", icon: "⚡" },
          ].map(item => (
            <div key={item.label} className={cx(
              "rounded-xl p-4 border transition-colors",
              item.ok ? "bg-white/[0.03] border-white/8" : "bg-white/[0.01] border-white/5"
            )}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-base">{item.icon}</span>
                <span className="text-xs text-neutral-500">{item.label}</span>
                <div className={cx("w-1.5 h-1.5 rounded-full ml-auto", item.ok ? "bg-green-400" : "bg-neutral-700")} />
              </div>
              <div className={cx("text-sm font-semibold", item.ok ? "text-white" : "text-neutral-600")}>{item.value}</div>
              <div className="text-[10px] text-neutral-600 mt-0.5">{item.sub}</div>
            </div>
          ))}
        </div>

        {/* Ollama section */}
        <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-xl">🦙</div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">Ollama</span>
                <span className="text-[10px] text-green-400 border border-green-500/20 px-2 py-0.5 rounded-full bg-green-500/8">FREE</span>
              </div>
              <p className="text-xs text-neutral-500 mt-0.5">Local or remote — supports API key for secured endpoints</p>
            </div>
            <div className={cx("flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border",
              ollamaOk ? "text-green-400 border-green-500/25 bg-green-500/8" : "text-neutral-600 border-white/8")}>
              <div className={cx("w-1.5 h-1.5 rounded-full", ollamaOk ? "bg-green-400 animate-pulse" : "bg-neutral-700")} />
              {ollamaOk ? `${ollamaStatus?.model_count} models · ${ollamaStatus?.latency_ms}ms` : "Offline"}
            </div>
          </div>

          <div className="space-y-2">
            <input value={ollamaUrl} onChange={e => setOllamaUrl(e.target.value)}
              placeholder="http://localhost:11434"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/25 focus:outline-none focus:border-cyan-500/40 font-mono" />
            <div className="flex gap-2">
              <input id="ollama-key-input" type="password"
                placeholder="API key (optional — for secured/remote Ollama)"
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/25 focus:outline-none focus:border-cyan-500/40 font-mono" />
              <button onClick={async () => {
                  const ki = document.getElementById("ollama-key-input") as HTMLInputElement;
                  const r = await apiFetch(`/settings/ollama?url=${encodeURIComponent(ollamaUrl)}&api_key=${encodeURIComponent(ki?.value||"")}`, { method: "POST" });
                  await load(); if (ki) ki.value = "";
                  if (r.available) toast.success(`Connected — ${r.model_count} models`);
                  else toast.error(r.error || "Cannot connect");
                }}
                className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-semibold rounded-xl transition-colors whitespace-nowrap">
                Save
              </button>
            </div>
          </div>

          {ollamaStatus?.available && ollamaStatus.models?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {ollamaStatus.models.slice(0, 8).map((m: any) => (
                <span key={m.name} className="text-[10px] bg-white/5 text-neutral-500 px-2 py-0.5 rounded-full border border-white/8 font-mono">
                  {m.name}
                </span>
              ))}
              {ollamaStatus.models.length > 8 && (
                <span className="text-[10px] text-neutral-600">+{ollamaStatus.models.length - 8} more</span>
              )}
            </div>
          )}

          {ollamaStatus?.error && (
            <p className="mt-2 text-xs text-amber-600/70">{ollamaStatus.error}</p>
          )}
        </div>

        {/* Cloud provider cards */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-white">Cloud Providers</h2>
            <span className="text-xs text-neutral-600">{configured.length} of {providerList.length} configured</span>
          </div>
          <div className="space-y-3">
            {Object.entries(keys).map(([pid, info]: [string, any]) => (
              <ProviderConnectionCard key={pid}
                info={info} pid={pid}
                editVal={editing[pid] || ""}
                isTesting={testing[pid]}
                testResult={testResults[pid]}
                onEdit={(v: string) => setEditing(prev => ({ ...prev, [pid]: v }))}
                onSave={() => saveKey(pid)}
                onTest={() => testKey(pid)}
                onRemove={() => removeKey(pid)}
                saving={saving}
                googleStatus={googleStatus}
                googleSetupStep={googleSetupStep}
                setGoogleSetupStep={setGoogleSetupStep}
                googleClientId={googleClientId}
                setGoogleClientId={setGoogleClientId}
                googleClientSecret={googleClientSecret}
                setGoogleClientSecret={setGoogleClientSecret}
                saveGoogleCreds={saveGoogleCreds}
                startGoogleOAuth={startGoogleOAuth}
                revokeGoogle={revokeGoogle}
                testGoogle={testGoogle}
                openAIStatus={openAIStatus}
                openAIGuide={openAIGuide}
                openAIKey={openAIKey}
                setOpenAIKey={setOpenAIKey}
                connectOpenAI={connectOpenAI}
              />
            ))}
          </div>
        </div>

        {/* .env hint */}
        <div className="rounded-xl border border-white/5 px-4 py-3">
          <p className="text-xs text-neutral-700">
            Keys stored in <code className="text-neutral-600 font-mono">.env</code> — take effect immediately, no restart needed.
          </p>
        </div>

      </div>
    </div>
  );
}


export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [apiReady, setApiReady] = useState<boolean | null>(null); // null=checking
  const [retries, setRetries] = useState(0);

  useEffect(() => {
    let attempts = 0;
    const tryConnect = () => {
      attempts++;
      setRetries(attempts);
      apiFetch("/health")
        .then(() => {
          setApiReady(true);
          // Load data once connected
          apiFetch("/agents").then(setAgents).catch(() => {});
          apiFetch("/stats").then(setStats).catch(() => {});
        })
        .catch(() => {
          setApiReady(false);
          setTimeout(tryConnect, 2000); // retry every 2s
        });
    };
    tryConnect();
    const iv = setInterval(() => {
      apiFetch("/agents").then(setAgents).catch(() => {});
      apiFetch("/stats").then(setStats).catch(() => {});
    }, 30000);
    return () => clearInterval(iv);
  }, []);

  // Loading / connecting screen
  if (!apiReady) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-neutral-950 text-white gap-6">
        <div className="text-center">
          <div className="text-5xl mb-4">🦅</div>
          <h1 className="text-xl font-bold text-cyan-400 mb-1">AgentFlow v2</h1>
          <p className="text-neutral-500 text-sm">
            {apiReady === null || retries <= 1
              ? "Connecting to API…"
              : `Waiting for API · attempt ${retries}…`}
          </p>
          <p className="text-neutral-700 text-xs mt-2 font-mono">{API}/health</p>
        </div>
        <div className="flex gap-1.5">
          {[0,1,2].map(i => (
            <div key={i} className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce"
              style={{ animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
        <div className="text-xs text-neutral-700 max-w-xs text-center">
          If this persists, make sure the server is running:<br />
          <span className="font-mono text-neutral-600">python3 server.py</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-neutral-950 text-white overflow-hidden" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
      <Toaster position="top-right" toastOptions={{ style: { background: "#171717", color: "#e5e5e5", border: "1px solid #262626", fontSize: "13px" } }} />
      <NavSidebar page={page} setPage={setPage} stats={stats} />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {page === "chat"      && <ChatPage agents={agents} />}
        {page === "models"    && <ModelsPage />}
        {page === "agents"    && <AgentsPage agents={agents} />}
        {page === "mcp"       && <MCPPage />}
        {page === "knowledge" && <KnowledgePage />}
        {page === "openclaw"  && <OpenClawPage />}
        {page === "settings"  && <SettingsPage stats={stats} />}
      </div>
    </div>
  );
}

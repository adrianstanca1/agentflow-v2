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
const API = (import.meta as any).env?.VITE_API_URL || "http://localhost:8000";
const WS_URL = (import.meta as any).env?.VITE_WS_URL || "ws://localhost:8000";

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
  session_id?: string; metadata?: Record<string, any>;
}
interface Message {
  id: string; role: "user" | "agent"; content: string;
  events: StreamEvent[]; agent: string; ts: Date; status: "streaming" | "done" | "error";
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
  const [preferLocal, setPreferLocal] = useState(false);
  const [sideOpen, setSideOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const meta = AGENT_META[agentKey] || AGENT_META.assistant;

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

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
      const body: any = { agent_key: agentKey, task, stream: true, prefer_local: preferLocal };
      if (model) body.model_override = model;
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
            if ((evt.event_type === "output" || evt.event_type === "complete") && typeof evt.content === "string" && evt.content)
              finalContent = evt.content;
            setMessages(prev => prev.map(m => m.id === aid ? { ...m, content: finalContent || m.content, events: [...evts],
              status: evt.event_type === "complete" ? "done" : "streaming" } : m));
          } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        toast.error(e.message);
        setMessages(prev => prev.map(m => m.id === aid ? { ...m, content: `Error: ${e.message}`, status: "error" } : m));
      }
    } finally { setRunning(false); abortRef.current = null; inputRef.current?.focus(); }
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

        <div className="border-t border-neutral-800 p-3">
          <div className="max-w-2xl mx-auto">
            <div className="flex gap-2 items-end bg-neutral-900 border border-neutral-700 rounded-xl px-3 py-2.5 focus-within:border-neutral-500 transition-colors">
              <textarea ref={inputRef} value={input} rows={1}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder={`Message ${agentKey.replace("_", " ")} agent...`}
                className="flex-1 bg-transparent text-sm text-neutral-200 placeholder-neutral-600 resize-none focus:outline-none leading-relaxed"
                style={{ minHeight: "22px", maxHeight: "160px" }} />
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
            <p className="text-[10px] text-neutral-700 mt-1.5 text-center">Enter to send · Shift+Enter newline</p>
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
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3">
      <div className={cx("w-8 h-8 rounded-xl bg-gradient-to-br flex items-center justify-center text-white flex-shrink-0 mt-0.5", meta.color)}>
        {meta.icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-xs font-semibold text-neutral-400 capitalize">{msg.agent.replace("_", " ")}</span>
          <span className="text-[10px] text-neutral-700">{msg.ts.toLocaleTimeString()}</span>
          {msg.status === "streaming" && <span className="flex items-center gap-1 text-[10px] text-neutral-500"><span className="w-1.5 h-1.5 rounded-full bg-neutral-500 animate-pulse" />generating</span>}
          {msg.status === "error" && <span className="text-[10px] text-red-500">error</span>}
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
  const [model, setModel] = useState("qwen2.5-coder:7b");
  const [cwd, setCwd] = useState(".");
  const [sessionId] = useState(() => "oc-" + Math.random().toString(36).slice(2, 8));
  const [task, setTask] = useState("");
  const [events, setEvents] = useState<Array<{type: string; content: any; ts: number}>>([]);
  const [running, setRunning] = useState(false);
  const [models, setModels] = useState<Array<{name: string; installed: boolean; recommended: boolean}>>([]);
  const [files, setFiles] = useState<Array<{name: string; path: string; is_dir: boolean; size: number}>>([]);
  const [openFile, setOpenFile] = useState<{path: string; content: string; extension: string; lines?: number} | null>(null);
  const [activeTab, setActiveTab] = useState<"agent" | "files" | "editor">("agent");
  const abortRef = useRef<AbortController | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    apiFetch("/openclaw/models").then(setModels).catch(() => {});
  }, []);

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

  const installed = models.filter(m => m.installed);

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
        body: JSON.stringify({ task: taskText, model, cwd, session_id: sessionId, stream: true }),
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
            <select value={model} onChange={e => setModel(e.target.value)}
              className="bg-neutral-900 border border-neutral-800 text-[11px] text-neutral-400 rounded px-2 py-1 focus:outline-none">
              {installed.length > 0
                ? installed.map(m => <option key={m.name} value={m.name}>{m.name}{m.recommended ? " ⭐" : ""}</option>)
                : <option>qwen2.5-coder:7b (not installed)</option>}
            </select>
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
                      {installed.length > 0
                        ? `Using ${model} • ${installed.length} models available`
                        : "Pull qwen2.5-coder:7b in Model Hub to start"}
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
                  <button onClick={run} disabled={!task.trim() || installed.length === 0}
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



function SettingsPage({ stats }: { stats: any }) {
  const [model, setModel] = useState(""); const [local, setLocal] = useState(false);
  const [providers, setProviders] = useState<CloudProvider[]>([]);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    apiFetch("/cloud/providers").then(setProviders).catch(() => {});
  }, []);

  const checkAll = async () => {
    setChecking(true);
    try {
      await apiFetch("/cloud/providers/check", { method: "POST" });
      const p = await apiFetch("/cloud/providers");
      setProviders(p);
      toast.success("Provider check complete");
    } catch (e: any) { toast.error(e.message); }
    finally { setChecking(false); }
  };

  const switchAll = async () => {
    if (!model) return;
    try { await apiFetch("/agents/switch-model", { method: "POST", body: JSON.stringify({ model, prefer_local: local }) }); toast.success(`All agents → ${model}`); }
    catch (e: any) { toast.error(e.message); }
  };
  const services = [
    { name: "AgentFlow API", url: "http://localhost:8000/docs" },
    { name: "Langfuse", url: "http://localhost:3001" },
    { name: "Temporal UI", url: "http://localhost:8080" },
    { name: "LiteLLM", url: "http://localhost:4000" },
    { name: "Open WebUI", url: "http://localhost:8888" },
    { name: "Qdrant", url: "http://localhost:6333/dashboard" },
  ];
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-xl mx-auto space-y-6">
        <h2 className="text-sm font-semibold text-white">Settings</h2>
        <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
          <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Platform Mode</p>
          <div className="flex items-center gap-3 mb-1"><div className={cx("w-2.5 h-2.5 rounded-full", stats?.mode === "hybrid" ? "bg-blue-400" : "bg-green-400")} /><p className="text-sm font-medium text-neutral-200 capitalize">{stats?.mode || "local"} Mode</p></div>
          <p className="text-xs text-neutral-600">{stats?.mode === "hybrid" ? "Cloud + local via LiteLLM" : "100% local — Ollama only, no API keys needed"}</p>
        </div>
        <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
          <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Switch All Agents</p>
          <input value={model} onChange={e => setModel(e.target.value)} placeholder="e.g. llama3.1:8b or claude-sonnet-4-6" className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-300 placeholder-neutral-600 focus:outline-none mb-3" />
          <div className="flex items-center gap-2 mb-3"><Toggle on={local} onChange={setLocal} /><span className="text-xs text-neutral-500">Use Ollama (local)</span></div>
          <button onClick={switchAll} className="px-4 py-2 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm text-white transition-colors">Switch All</button>
        </div>
        {stats?.ollama && (
          <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
            <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Ollama</p>
            <div className="grid grid-cols-3 gap-3">
              {[{label:"Hosts",value:stats.ollama.total_hosts},{label:"Healthy",value:stats.ollama.healthy_hosts},{label:"Models",value:stats.ollama.total_models}].map(s => (
                <div key={s.label} className="text-center p-2 rounded-lg bg-neutral-800"><p className="text-xl font-bold text-white">{s.value}</p><p className="text-xs text-neutral-600">{s.label}</p></div>
              ))}
            </div>
          </div>
        )}
        {/* Cloud providers */}
        <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Cloud Providers</p>
            <button onClick={checkAll} disabled={checking}
              className="flex items-center gap-1.5 text-xs text-neutral-500 hover:text-neutral-300 transition-colors">
              {checking ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
              Test keys
            </button>
          </div>
          <div className="space-y-1.5">
            {providers.map(p => (
              <div key={p.id} className="flex items-center gap-3 py-1.5">
                <span className="text-base w-6 text-center flex-shrink-0">{p.icon}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-neutral-300">{p.name}</p>
                  <p className="text-[10px] font-mono text-neutral-600">{p.id.toUpperCase()}_API_KEY</p>
                </div>
                <div className="flex-shrink-0">
                  {p.configured ? (
                    p.healthy === null ? (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-500">unchecked</span>
                    ) : p.healthy ? (
                      <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-400" /> valid
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> invalid
                      </span>
                    )
                  ) : (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-600 border border-amber-500/20">not set</span>
                  )}
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-neutral-700 mt-3">Set keys in <code className="text-neutral-600">.env</code> and restart to enable cloud models</p>
        </div>

        <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800">
          <p className="text-xs font-medium text-neutral-500 mb-3 uppercase tracking-wider">Services</p>
          <div className="space-y-1">
            {services.map(s => (
              <a key={s.name} href={s.url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-3 py-2 px-2 -mx-2 hover:bg-neutral-800 rounded-lg transition-colors group">
                <div className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0" />
                <p className="text-sm text-neutral-400 group-hover:text-neutral-200 transition-colors flex-1">{s.name}</p>
                <p className="text-xs text-neutral-700 font-mono">{s.url.replace("http://localhost:", ":")}</p>
                <ArrowRight className="w-3 h-3 text-neutral-700 group-hover:text-neutral-500" />
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Root App
export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    const load = () => {
      apiFetch("/agents").then(setAgents).catch(() => {});
      apiFetch("/stats").then(setStats).catch(() => {});
    };
    load();
    const iv = setInterval(load, 30000);
    return () => clearInterval(iv);
  }, []);

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
        {page === "settings"  && <SettingsPage stats={stats} />}
      </div>
    </div>
  );
}

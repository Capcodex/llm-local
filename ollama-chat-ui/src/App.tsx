import { useEffect, useMemo, useRef, useState } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
}

interface ChatThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
}

interface ToolCall {
  name: string;
  args: Record<string, unknown>;
}

interface PendingConfirmation {
  threadId: string;
  toolCall: ToolCall;
  originalMessage: string;
}

interface ChatApiResponse {
  reply: string;
  requires_confirmation?: boolean;
  tool_call?: ToolCall;
}

const THREAD_STORAGE_KEY = 'chatgpt_like_threads_v1';
const ACTIVE_THREAD_STORAGE_KEY = 'chatgpt_like_active_thread_v1';

const TOOL_OPTIONS = [
  { id: 'get_current_time', label: 'Heure actuelle' },
  { id: 'create_note', label: 'Creer une note' },
];

const STARTER_PROMPTS = [
  'Explique-moi une idee complexe de maniere simple.',
  'Aide-moi a structurer une roadmap produit en 3 phases.',
  'Propose un plan d action pour apprendre LangChain en 30 jours.',
  'Reformule ce texte pour un email professionnel et concis.',
];

function createId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function createEmptyThread(): ChatThread {
  const now = new Date().toISOString();
  return {
    id: createId(),
    title: 'Nouvelle conversation',
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function deriveTitle(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) {
    return 'Nouvelle conversation';
  }
  if (trimmed.length <= 48) {
    return trimmed;
  }
  return `${trimmed.slice(0, 45)}...`;
}

function formatClock(isoDate: string): string {
  return new Date(isoDate).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function renderTextSegment(segment: string): React.ReactNode {
  const lines = segment.split('\n');
  return lines.map((line, index) => (
    <span key={`line-${index}`}>
      {line}
      {index < lines.length - 1 ? <br /> : null}
    </span>
  ));
}

function MessageContent({ content }: { content: string }) {
  const elements: React.ReactNode[] = [];
  const codeRegex = /```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null = null;

  while ((match = codeRegex.exec(content)) !== null) {
    const fullMatch = match[0];
    const language = match[1] || 'text';
    const code = match[2] || '';
    const textBefore = content.slice(lastIndex, match.index);

    if (textBefore.trim()) {
      elements.push(
        <p key={`text-${lastIndex}`} className="message-text">
          {renderTextSegment(textBefore.trim())}
        </p>,
      );
    }

    elements.push(
      <div className="code-block" key={`code-${lastIndex}`}>
        <div className="code-language">{language}</div>
        <pre>
          <code>{code}</code>
        </pre>
      </div>,
    );

    lastIndex = match.index + fullMatch.length;
  }

  const trailing = content.slice(lastIndex);
  if (trailing.trim()) {
    elements.push(
      <p key={`text-trailing-${lastIndex}`} className="message-text">
        {renderTextSegment(trailing.trim())}
      </p>,
    );
  }

  if (elements.length === 0) {
    return <p className="message-text" />;
  }

  return <>{elements}</>;
}

function App() {
  const [threads, setThreads] = useState<ChatThread[]>(() => {
    try {
      const raw = localStorage.getItem(THREAD_STORAGE_KEY);
      if (!raw) {
        return [createEmptyThread()];
      }
      const parsed = JSON.parse(raw) as ChatThread[];
      if (!Array.isArray(parsed) || parsed.length === 0) {
        return [createEmptyThread()];
      }
      return parsed;
    } catch {
      return [createEmptyThread()];
    }
  });

  const [activeThreadId, setActiveThreadId] = useState<string>(() => {
    const existing = localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY);
    return existing || '';
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
  const [enabledTools, setEnabledTools] = useState<Record<string, boolean>>({
    get_current_time: true,
    create_note: true,
  });

  const requireToolConfirmation = true;
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    localStorage.setItem(THREAD_STORAGE_KEY, JSON.stringify(threads));
  }, [threads]);

  useEffect(() => {
    if (!activeThreadId && threads.length > 0) {
      setActiveThreadId(threads[0].id);
      return;
    }

    const stillExists = threads.some((thread) => thread.id === activeThreadId);
    if (!stillExists && threads.length > 0) {
      setActiveThreadId(threads[0].id);
    }
  }, [activeThreadId, threads]);

  useEffect(() => {
    if (activeThreadId) {
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, activeThreadId);
    }
  }, [activeThreadId]);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? threads[0],
    [activeThreadId, threads],
  );

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [activeThread, isLoading, pendingConfirmation]);

  const updateThread = (threadId: string, updater: (thread: ChatThread) => ChatThread) => {
    setThreads((prev) => prev.map((thread) => (thread.id === threadId ? updater(thread) : thread)));
  };

  const createAssistantMessage = (content: string): Message => ({
    id: createId(),
    role: 'assistant',
    content,
    createdAt: new Date().toISOString(),
  });

  const sendMessage = async (message: string, conversation: Message[], approvals: ToolCall[] = []) => {
    const response = await fetch('http://localhost:5001/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        messages: conversation.map((item) => ({ role: item.role, content: item.content })),
        enabled_tools: TOOL_OPTIONS.filter((tool) => enabledTools[tool.id]).map((tool) => tool.id),
        require_tool_confirmation: requireToolConfirmation,
        tool_approvals: approvals,
      }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => null);
      const serverMessage =
        errorPayload?.details || errorPayload?.error || `HTTP error! status: ${response.status}`;
      throw new Error(serverMessage);
    }

    return (await response.json()) as ChatApiResponse;
  };

  const appendAssistantReply = (threadId: string, content: string) => {
    updateThread(threadId, (thread) => ({
      ...thread,
      updatedAt: new Date().toISOString(),
      messages: [...thread.messages, createAssistantMessage(content)],
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || pendingConfirmation || !activeThread) {
      return;
    }

    const messageText = input.trim();
    const userMessage: Message = {
      id: createId(),
      role: 'user',
      content: messageText,
      createdAt: new Date().toISOString(),
    };

    const conversationForBackend = [...activeThread.messages, userMessage];
    const threadId = activeThread.id;

    setInput('');
    setIsLoading(true);

    updateThread(threadId, (thread) => {
      const shouldUpdateTitle =
        thread.title === 'Nouvelle conversation' && !thread.messages.some((msg) => msg.role === 'user');
      return {
        ...thread,
        title: shouldUpdateTitle ? deriveTitle(messageText) : thread.title,
        updatedAt: new Date().toISOString(),
        messages: [...thread.messages, userMessage],
      };
    });

    try {
      const data = await sendMessage(messageText, conversationForBackend);
      appendAssistantReply(threadId, data.reply);

      if (data.requires_confirmation && data.tool_call) {
        setPendingConfirmation({
          threadId,
          toolCall: data.tool_call,
          originalMessage: messageText,
        });
      } else {
        setPendingConfirmation(null);
      }
    } catch (error) {
      const errorText =
        error instanceof Error && error.message
          ? `Erreur: ${error.message}`
          : "Erreur: impossible de contacter le serveur backend. Verifie qu'il tourne.";
      appendAssistantReply(threadId, errorText);
      setPendingConfirmation(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfirmTool = async () => {
    if (!pendingConfirmation) {
      return;
    }

    const targetThread = threads.find((thread) => thread.id === pendingConfirmation.threadId);
    if (!targetThread) {
      setPendingConfirmation(null);
      return;
    }

    setIsLoading(true);

    try {
      const data = await sendMessage(
        pendingConfirmation.originalMessage,
        targetThread.messages,
        [pendingConfirmation.toolCall],
      );

      appendAssistantReply(targetThread.id, data.reply);

      if (data.requires_confirmation && data.tool_call) {
        setPendingConfirmation({
          ...pendingConfirmation,
          toolCall: data.tool_call,
        });
      } else {
        setPendingConfirmation(null);
      }
    } catch (error) {
      const errorText =
        error instanceof Error && error.message
          ? `Erreur: ${error.message}`
          : "Erreur: impossible de contacter le serveur backend. Verifie qu'il tourne.";
      appendAssistantReply(targetThread.id, errorText);
      setPendingConfirmation(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelTool = () => {
    if (!pendingConfirmation) {
      return;
    }
    appendAssistantReply(pendingConfirmation.threadId, 'Action annulee.');
    setPendingConfirmation(null);
  };

  const handleCreateThread = () => {
    const newThread = createEmptyThread();
    setThreads((prev) => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
    setPendingConfirmation(null);
    setInput('');
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const handleDeleteThread = (threadId: string) => {
    setThreads((prev) => {
      const filtered = prev.filter((thread) => thread.id !== threadId);
      return filtered.length > 0 ? filtered : [createEmptyThread()];
    });

    if (pendingConfirmation?.threadId === threadId) {
      setPendingConfirmation(null);
    }
  };

  const handlePromptClick = (prompt: string) => {
    setInput(prompt);
    inputRef.current?.focus();
  };

  const pendingForActiveThread =
    pendingConfirmation && activeThread ? pendingConfirmation.threadId === activeThread.id : false;

  return (
    <div className="chat-app-shell">
      <aside className="sidebar">
        <button type="button" className="new-chat-button" onClick={handleCreateThread}>
          + Nouveau chat
        </button>

        <div className="thread-list">
          {threads.map((thread) => {
            const preview = thread.messages[thread.messages.length - 1]?.content || 'Conversation vide';
            const isActive = thread.id === activeThread?.id;
            return (
              <button
                type="button"
                key={thread.id}
                className={`thread-item ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveThreadId(thread.id);
                  setPendingConfirmation((current) =>
                    current && current.threadId !== thread.id ? null : current,
                  );
                }}
              >
                <div className="thread-title">{thread.title}</div>
                <div className="thread-preview">{preview}</div>
                <div className="thread-meta">
                  <span>{new Date(thread.updatedAt).toLocaleDateString()}</span>
                  <span>{formatClock(thread.updatedAt)}</span>
                </div>
                <span
                  className="thread-delete"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleDeleteThread(thread.id);
                  }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      handleDeleteThread(thread.id);
                    }
                  }}
                >
                  Suppr.
                </span>
              </button>
            );
          })}
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <h1>Assistant IA</h1>
            <p>
              Style conversationnel proche de ChatGPT, avec memoire locale, confirmations d actions et outils
              configurables.
            </p>
          </div>

          <div className="tool-toggles">
            {TOOL_OPTIONS.map((tool) => (
              <label key={tool.id} className={`tool-chip ${enabledTools[tool.id] ? 'enabled' : ''}`}>
                <input
                  type="checkbox"
                  checked={enabledTools[tool.id]}
                  onChange={() =>
                    setEnabledTools((prev) => ({
                      ...prev,
                      [tool.id]: !prev[tool.id],
                    }))
                  }
                  disabled={isLoading}
                />
                {tool.label}
              </label>
            ))}
          </div>
        </header>

        <section className="chat-area" ref={chatContainerRef}>
          {activeThread?.messages.length ? (
            activeThread.messages.map((msg) => (
              <article key={msg.id} className={`message-row ${msg.role}`}>
                <div className={`avatar ${msg.role}`}>{msg.role === 'assistant' ? 'AI' : 'Toi'}</div>
                <div className="message-card">
                  <MessageContent content={msg.content} />
                  <div className="message-meta">{formatClock(msg.createdAt)}</div>
                </div>
              </article>
            ))
          ) : (
            <div className="empty-state">
              <h2>Demarre une conversation</h2>
              <p>
                Ton agent garde le contexte du fil actif pour mieux enchainer les demandes, comme une vraie session
                ChatGPT.
              </p>
              <div className="starter-prompts">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    type="button"
                    key={prompt}
                    className="prompt-card"
                    onClick={() => handlePromptClick(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {isLoading ? (
            <article className="message-row assistant">
              <div className="avatar assistant">AI</div>
              <div className="message-card typing-card">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </article>
          ) : null}

          {pendingForActiveThread && pendingConfirmation ? (
            <article className="confirmation-card">
              <strong>Confirmation requise</strong>
              <p>
                L outil <code>{pendingConfirmation.toolCall.name}</code> veut etre execute avec ces parametres:
              </p>
              <pre>{JSON.stringify(pendingConfirmation.toolCall.args, null, 2)}</pre>
              <div className="confirmation-actions">
                <button type="button" className="btn-approve" onClick={handleConfirmTool} disabled={isLoading}>
                  Confirmer
                </button>
                <button type="button" className="btn-deny" onClick={handleCancelTool} disabled={isLoading}>
                  Refuser
                </button>
              </div>
            </article>
          ) : null}
        </section>

        <footer className="composer">
          <form onSubmit={handleSubmit}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Envoie un message... (Entree pour envoyer, Shift+Entree pour sauter une ligne)"
              disabled={isLoading || pendingForActiveThread}
              rows={2}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void handleSubmit(event);
                }
              }}
            />
            <button type="submit" disabled={isLoading || pendingForActiveThread || !input.trim()}>
              Envoyer
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}

export default App;

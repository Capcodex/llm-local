import { useEffect, useMemo, useRef, useState } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  sources?: RagSource[];
  usedRag?: boolean;
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
  sources?: RagSource[];
  used_rag?: boolean;
  retrieved_chunks?: number;
  top_k?: number;
}

interface RagDocument {
  doc_id: string;
  filename: string;
  stored_filename?: string;
  file_type?: string;
  size_bytes?: number;
  chunks?: number;
  created_at?: string;
}

interface RagSource {
  source_id: string;
  doc_id?: string;
  filename?: string;
  chunk_index?: number;
  score?: number;
  preview?: string;
}

interface RagDocumentsResponse {
  documents: RagDocument[];
  total: number;
}

interface RagUploadError {
  filename: string;
  error: string;
}

interface RagUploadResponse {
  added_documents: RagDocument[];
  errors: RagUploadError[];
  total_uploaded: number;
  total_added: number;
  total_failed: number;
  error?: string;
}

// Sprint 6 types
type ViewType = 'chat' | 'missions' | 'admin';
type AdminTabType = 'missions' | 'tools' | 'skills' | 'logs' | 'registry';

interface Mission {
  id: string;
  title?: string;
  prompt: string;
  autonomy_level: string;
  status: string;
  risk_level?: string;
  started_at?: string;
  ended_at?: string;
  steps?: MissionStep[];
}

interface MissionStep {
  id: string;
  node_name: string;
  status: string;
  input_payload?: Record<string, unknown>;
  output_payload?: Record<string, unknown>;
  started_at?: string;
  ended_at?: string;
}

interface AdminTool {
  id: string;
  name: string;
  slug: string;
  version: string;
  description?: string;
  file_path?: string;
  status: string;
  created_by_mission_id?: string;
  created_at?: string;
  test_runs?: AdminTestRun[];
  registry_entries?: AdminRegistryEntry[];
}

interface AdminTestRun {
  id: string;
  attempt_number: number;
  status: string;
  stdout?: string;
  stderr?: string;
  traceback?: string;
  created_at?: string;
}

interface AdminRegistryEntry {
  id: string;
  published_version: string;
  validation_status: string;
  published_at?: string;
}

interface AdminSkill {
  id: string;
  tool_id?: string;
  title: string;
  slug: string;
  summary?: string;
  status: string;
  created_at?: string;
  markdown_content?: string;
}

interface AuditLogEntry {
  id: string;
  actor_type: string;
  action: string;
  target_type?: string;
  target_id?: string;
  metadata?: Record<string, unknown>;
  created_at?: string;
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

function formatDate(isoDate?: string): string {
  if (!isoDate) {
    return '';
  }
  const parsed = new Date(isoDate);
  if (Number.isNaN(parsed.getTime())) {
    return '';
  }
  return parsed.toLocaleDateString();
}

function formatBytes(size?: number): string {
  if (!size || size <= 0) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = size;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatRelevanceScore(score?: number): string {
  if (typeof score !== 'number' || Number.isNaN(score)) {
    return 'n/a';
  }
  return score.toFixed(2);
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

// ─── Status badge helpers ────────────────────────────────────────────────────

function missionStatusBadge(status: string): React.ReactNode {
  const map: Record<string, string> = {
    pending: 'badge bg-secondary',
    planning: 'badge bg-primary',
    forging: 'badge bg-primary',
    testing: 'badge bg-primary',
    done: 'badge bg-success',
    error: 'badge bg-danger',
    awaiting_approval: 'badge bg-warning text-dark',
  };
  const cls = map[status] ?? 'badge bg-secondary';
  return <span className={cls}>{status}</span>;
}

function toolStatusBadge(status: string): React.ReactNode {
  const map: Record<string, string> = {
    active: 'badge bg-success',
    disabled: 'badge bg-warning text-dark',
    archived: 'badge bg-danger',
    candidate: 'badge bg-warning text-dark',
    pending: 'badge bg-secondary',
  };
  const cls = map[status] ?? 'badge bg-secondary';
  return <span className={cls}>{status}</span>;
}

function skillStatusBadge(status: string): React.ReactNode {
  const map: Record<string, string> = {
    active: 'badge bg-success',
    draft: 'badge bg-secondary',
    archived: 'badge bg-danger',
  };
  const cls = map[status] ?? 'badge bg-secondary';
  return <span className={cls}>{status}</span>;
}

function validationStatusBadge(status: string): React.ReactNode {
  const map: Record<string, string> = {
    pending: 'badge bg-warning text-dark',
    approved: 'badge bg-success',
    rejected: 'badge bg-danger',
  };
  const cls = map[status] ?? 'badge bg-secondary';
  return <span className={cls}>{status}</span>;
}

const API_BASE = 'http://localhost:5001';

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
  const [documents, setDocuments] = useState<RagDocument[]>([]);
  const [isDocumentsLoading, setIsDocumentsLoading] = useState(false);
  const [isUploadingDocs, setIsUploadingDocs] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [documentsStatus, setDocumentsStatus] = useState<string | null>(null);
  const [useRag, setUseRag] = useState(false);

  // Sprint 6 — routing state
  const [currentView, setCurrentView] = useState<ViewType>('chat');
  const [adminTab, setAdminTab] = useState<AdminTabType>('missions');

  // Sprint 6 — Missions view state
  const [missions, setMissions] = useState<Mission[]>([]);
  const [missionsLoading, setMissionsLoading] = useState(false);
  const [missionsError, setMissionsError] = useState<string | null>(null);
  const [expandedMissionId, setExpandedMissionId] = useState<string | null>(null);
  const [newMissionPrompt, setNewMissionPrompt] = useState('');
  const [newMissionAutonomy, setNewMissionAutonomy] = useState<string>('supervised');
  const [missionCreateLoading, setMissionCreateLoading] = useState(false);
  const [missionCreateError, setMissionCreateError] = useState<string | null>(null);
  const [missionActionLoading, setMissionActionLoading] = useState<string | null>(null);

  // Sprint 6 — Admin Tools state
  const [adminTools, setAdminTools] = useState<AdminTool[]>([]);
  const [adminToolsLoading, setAdminToolsLoading] = useState(false);
  const [adminToolsError, setAdminToolsError] = useState<string | null>(null);
  const [expandedToolId, setExpandedToolId] = useState<string | null>(null);
  const [toolActionLoading, setToolActionLoading] = useState<string | null>(null);

  // Sprint 6 — Admin Skills state
  const [adminSkills, setAdminSkills] = useState<AdminSkill[]>([]);
  const [adminSkillsLoading, setAdminSkillsLoading] = useState(false);
  const [adminSkillsError, setAdminSkillsError] = useState<string | null>(null);
  const [expandedSkillId, setExpandedSkillId] = useState<string | null>(null);

  // Sprint 6 — Admin Logs state
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [auditLogsLoading, setAuditLogsLoading] = useState(false);
  const [auditLogsError, setAuditLogsError] = useState<string | null>(null);
  const [logsActionFilter, setLogsActionFilter] = useState('');

  // Sprint 6 — Admin Registry state (reuses missions with awaiting_approval)
  const [registryMissions, setRegistryMissions] = useState<Mission[]>([]);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [registryActionLoading, setRegistryActionLoading] = useState<string | null>(null);

  // Admin missions sub-tab (reuses missions state)
  const [adminMissions, setAdminMissions] = useState<Mission[]>([]);
  const [adminMissionsLoading, setAdminMissionsLoading] = useState(false);
  const [adminMissionsError, setAdminMissionsError] = useState<string | null>(null);
  const [expandedAdminMissionId, setExpandedAdminMissionId] = useState<string | null>(null);

  const requireToolConfirmation = true;
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
  const latestDocumentDate = useMemo(() => {
    if (documents.length === 0) {
      return '';
    }
    const latest = documents
      .map((doc) => doc.created_at || '')
      .filter((value) => value)
      .sort()
      .at(-1);
    return latest ? formatDate(latest) : '';
  }, [documents]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [activeThread, isLoading, pendingConfirmation]);

  useEffect(() => {
    void fetchDocuments();
  }, []);

  // Fetch missions when missions view is shown
  useEffect(() => {
    if (currentView === 'missions') {
      void fetchMissions();
    }
  }, [currentView]);

  // Fetch admin sub-tab data when tab changes
  useEffect(() => {
    if (currentView === 'admin') {
      if (adminTab === 'missions') void fetchAdminMissions();
      if (adminTab === 'tools') void fetchAdminTools();
      if (adminTab === 'skills') void fetchAdminSkills();
      if (adminTab === 'logs') void fetchAuditLogs();
      if (adminTab === 'registry') void fetchRegistryMissions();
    }
  }, [currentView, adminTab]);

  const updateThread = (threadId: string, updater: (thread: ChatThread) => ChatThread) => {
    setThreads((prev) => prev.map((thread) => (thread.id === threadId ? updater(thread) : thread)));
  };

  const createAssistantMessage = (content: string, sources?: RagSource[], usedRag?: boolean): Message => ({
    id: createId(),
    role: 'assistant',
    content,
    createdAt: new Date().toISOString(),
    sources,
    usedRag,
  });

  const fetchDocuments = async () => {
    setIsDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const response = await fetch('http://localhost:5001/rag/documents');
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.error || `Erreur HTTP ${response.status}`);
      }
      const payload = (await response.json()) as RagDocumentsResponse;
      setDocuments(Array.isArray(payload.documents) ? payload.documents : []);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Impossible de charger la liste des documents.';
      setDocumentsError(message);
    } finally {
      setIsDocumentsLoading(false);
    }
  };

  const uploadDocuments = async (files: FileList) => {
    if (!files.length) {
      return;
    }
    setIsUploadingDocs(true);
    setDocumentsError(null);
    setDocumentsStatus(null);

    try {
      const formData = new FormData();
      Array.from(files).forEach((file) => formData.append('files', file));

      const response = await fetch('http://localhost:5001/rag/documents', {
        method: 'POST',
        body: formData,
      });

      const payload = (await response.json().catch(() => null)) as RagUploadResponse | null;
      if (!payload) {
        throw new Error(`Upload echoue (HTTP ${response.status}).`);
      }

      const detailedErrors =
        Array.isArray(payload.errors) && payload.errors.length > 0
          ? payload.errors.map((item) => `${item.filename}: ${item.error}`).join(' | ')
          : '';

      if (!response.ok) {
        throw new Error(payload.error || detailedErrors || `Upload echoue (HTTP ${response.status}).`);
      }

      const statusParts = [
        `${payload.total_added} ajoute(s)`,
        `${payload.total_failed} erreur(s)`,
      ];
      setDocumentsStatus(statusParts.join(' - '));

      if (detailedErrors) {
        setDocumentsError(detailedErrors);
      }
      await fetchDocuments();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "L'import des documents a echoue.";
      setDocumentsError(message);
    } finally {
      setIsUploadingDocs(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const deleteDocument = async (docId: string) => {
    try {
      const response = await fetch(`http://localhost:5001/rag/documents/${docId}`, {
        method: 'DELETE',
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(payload?.error || `Suppression echouee (HTTP ${response.status}).`);
      }
      setDocumentsStatus('Document supprime.');
      await fetchDocuments();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Impossible de supprimer ce document.';
      setDocumentsError(message);
    }
  };

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
        use_rag: useRag,
        rag_top_k: 4,
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

  const appendAssistantReply = (
    threadId: string,
    content: string,
    options?: { sources?: RagSource[]; usedRag?: boolean },
  ) => {
    updateThread(threadId, (thread) => ({
      ...thread,
      updatedAt: new Date().toISOString(),
      messages: [
        ...thread.messages,
        createAssistantMessage(content, options?.sources, options?.usedRag),
      ],
    }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement | HTMLTextAreaElement>) => {
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
      appendAssistantReply(threadId, data.reply, {
        sources: data.sources,
        usedRag: data.used_rag,
      });

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

      appendAssistantReply(targetThread.id, data.reply, {
        sources: data.sources,
        usedRag: data.used_rag,
      });

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

  const handleOpenFilePicker = () => {
    fileInputRef.current?.click();
  };

  const handleFilesSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) {
      return;
    }
    await uploadDocuments(event.target.files);
  };

  // ─── Sprint 6 — Missions API ────────────────────────────────────────────────

  const fetchMissions = async () => {
    setMissionsLoading(true);
    setMissionsError(null);
    try {
      const res = await fetch(`${API_BASE}/missions`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setMissions(Array.isArray(data) ? data : (data.missions ?? []));
    } catch (err) {
      setMissionsError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setMissionsLoading(false);
    }
  };

  const createMission = async () => {
    if (!newMissionPrompt.trim()) return;
    setMissionCreateLoading(true);
    setMissionCreateError(null);
    try {
      const res = await fetch(`${API_BASE}/missions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: newMissionPrompt.trim(), autonomy_level: newMissionAutonomy }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      setNewMissionPrompt('');
      await fetchMissions();
    } catch (err) {
      setMissionCreateError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setMissionCreateLoading(false);
    }
  };

  const runMission = async (id: string) => {
    setMissionActionLoading(id + ':run');
    try {
      await fetch(`${API_BASE}/missions/${id}/run`, { method: 'POST' });
      await fetchMissions();
    } finally {
      setMissionActionLoading(null);
    }
  };

  const approveMission = async (id: string) => {
    setMissionActionLoading(id + ':approve');
    try {
      await fetch(`${API_BASE}/missions/${id}/approve`, { method: 'POST' });
      await fetchMissions();
    } finally {
      setMissionActionLoading(null);
    }
  };

  const rejectMission = async (id: string) => {
    setMissionActionLoading(id + ':reject');
    try {
      await fetch(`${API_BASE}/missions/${id}/reject`, { method: 'POST' });
      await fetchMissions();
    } finally {
      setMissionActionLoading(null);
    }
  };

  // ─── Sprint 6 — Admin Missions API ─────────────────────────────────────────

  const fetchAdminMissions = async () => {
    setAdminMissionsLoading(true);
    setAdminMissionsError(null);
    try {
      const res = await fetch(`${API_BASE}/missions`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setAdminMissions(Array.isArray(data) ? data : (data.missions ?? []));
    } catch (err) {
      setAdminMissionsError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setAdminMissionsLoading(false);
    }
  };

  // ─── Sprint 6 — Admin Tools API ────────────────────────────────────────────

  const fetchAdminTools = async () => {
    setAdminToolsLoading(true);
    setAdminToolsError(null);
    try {
      const res = await fetch(`${API_BASE}/tools`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setAdminTools(Array.isArray(data) ? data : (data.tools ?? []));
    } catch (err) {
      setAdminToolsError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setAdminToolsLoading(false);
    }
  };

  const patchToolStatus = async (id: string, status: string) => {
    setToolActionLoading(id + ':' + status);
    try {
      await fetch(`${API_BASE}/tools/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      await fetchAdminTools();
    } finally {
      setToolActionLoading(null);
    }
  };

  // ─── Sprint 6 — Admin Skills API ───────────────────────────────────────────

  const fetchAdminSkills = async () => {
    setAdminSkillsLoading(true);
    setAdminSkillsError(null);
    try {
      const res = await fetch(`${API_BASE}/skills`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setAdminSkills(Array.isArray(data) ? data : (data.skills ?? []));
    } catch (err) {
      setAdminSkillsError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setAdminSkillsLoading(false);
    }
  };

  // ─── Sprint 6 — Audit Logs API ─────────────────────────────────────────────

  const fetchAuditLogs = async () => {
    setAuditLogsLoading(true);
    setAuditLogsError(null);
    try {
      const res = await fetch(`${API_BASE}/audit-logs`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setAuditLogs(Array.isArray(data) ? data : (data.logs ?? []));
    } catch (err) {
      setAuditLogsError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setAuditLogsLoading(false);
    }
  };

  // ─── Sprint 6 — Registry API ───────────────────────────────────────────────

  const fetchRegistryMissions = async () => {
    setRegistryLoading(true);
    setRegistryError(null);
    try {
      const res = await fetch(`${API_BASE}/missions`);
      if (!res.ok) {
        const payload = await res.json().catch(() => null);
        throw new Error(payload?.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const all: Mission[] = Array.isArray(data) ? data : (data.missions ?? []);
      setRegistryMissions(all.filter((m) => m.status === 'awaiting_approval'));
    } catch (err) {
      setRegistryError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setRegistryLoading(false);
    }
  };

  const approveRegistryMission = async (id: string) => {
    setRegistryActionLoading(id + ':approve');
    try {
      await fetch(`${API_BASE}/missions/${id}/approve`, { method: 'POST' });
      await fetchRegistryMissions();
    } finally {
      setRegistryActionLoading(null);
    }
  };

  const rejectRegistryMission = async (id: string) => {
    setRegistryActionLoading(id + ':reject');
    try {
      await fetch(`${API_BASE}/missions/${id}/reject`, { method: 'POST' });
      await fetchRegistryMissions();
    } finally {
      setRegistryActionLoading(null);
    }
  };

  const pendingForActiveThread =
    pendingConfirmation && activeThread ? pendingConfirmation.threadId === activeThread.id : false;

  // ─── Filtered audit logs ────────────────────────────────────────────────────
  const filteredAuditLogs = useMemo(() => {
    if (!logsActionFilter.trim()) return auditLogs;
    const q = logsActionFilter.trim().toLowerCase();
    return auditLogs.filter((log) => log.action.toLowerCase().includes(q));
  }, [auditLogs, logsActionFilter]);

  // ─── Admin sub-tab content renderers ───────────────────────────────────────

  const renderAdminMissions = () => (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Missions</h5>
        <button
          className="btn btn-sm btn-outline-secondary"
          onClick={() => void fetchAdminMissions()}
          disabled={adminMissionsLoading}
        >
          {adminMissionsLoading ? (
            <span className="spinner-border spinner-border-sm me-1" />
          ) : null}
          Rafraichir
        </button>
      </div>

      {adminMissionsError ? (
        <div className="alert alert-danger">{adminMissionsError}</div>
      ) : null}

      {adminMissionsLoading && adminMissions.length === 0 ? (
        <div className="text-center py-4">
          <span className="spinner-border text-secondary" />
        </div>
      ) : null}

      {!adminMissionsLoading && adminMissions.length === 0 ? (
        <p className="text-muted">Aucune mission.</p>
      ) : null}

      {adminMissions.length > 0 ? (
        <div className="table-responsive">
          <table className="table table-dark table-hover table-sm align-middle">
            <thead>
              <tr>
                <th>ID</th>
                <th>Prompt</th>
                <th>Status</th>
                <th>Autonomy</th>
                <th>Started At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {adminMissions.map((m) => (
                <>
                  <tr
                    key={m.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setExpandedAdminMissionId((prev) => (prev === m.id ? null : m.id))}
                  >
                    <td>
                      <code className="small">{m.id.slice(0, 8)}</code>
                    </td>
                    <td>
                      <span title={m.prompt}>
                        {m.prompt.length > 60 ? `${m.prompt.slice(0, 57)}...` : m.prompt}
                      </span>
                    </td>
                    <td>{missionStatusBadge(m.status)}</td>
                    <td>{m.autonomy_level}</td>
                    <td>{m.started_at ? formatDate(m.started_at) : '—'}</td>
                    <td onClick={(e) => e.stopPropagation()}>
                      {m.status === 'awaiting_approval' ? (
                        <>
                          <button
                            className="btn btn-sm btn-success me-1"
                            onClick={() => void approveMission(m.id)}
                            disabled={missionActionLoading !== null}
                          >
                            Approve
                          </button>
                          <button
                            className="btn btn-sm btn-danger"
                            onClick={() => void rejectMission(m.id)}
                            disabled={missionActionLoading !== null}
                          >
                            Reject
                          </button>
                        </>
                      ) : null}
                    </td>
                  </tr>
                  {expandedAdminMissionId === m.id && m.steps && m.steps.length > 0 ? (
                    <tr key={`${m.id}-steps`}>
                      <td colSpan={6}>
                        <div className="p-2">
                          <strong className="small">Steps:</strong>
                          <table className="table table-dark table-sm mt-1 mb-0">
                            <thead>
                              <tr>
                                <th>Node</th>
                                <th>Status</th>
                                <th>Started At</th>
                              </tr>
                            </thead>
                            <tbody>
                              {m.steps.map((step) => (
                                <tr key={step.id}>
                                  <td>{step.node_name}</td>
                                  <td>{missionStatusBadge(step.status)}</td>
                                  <td>{step.started_at ? formatDate(step.started_at) : '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </td>
                    </tr>
                  ) : null}
                </>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );

  const renderAdminTools = () => (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Tools</h5>
        <button
          className="btn btn-sm btn-outline-secondary"
          onClick={() => void fetchAdminTools()}
          disabled={adminToolsLoading}
        >
          {adminToolsLoading ? (
            <span className="spinner-border spinner-border-sm me-1" />
          ) : null}
          Rafraichir
        </button>
      </div>

      {adminToolsError ? (
        <div className="alert alert-danger">{adminToolsError}</div>
      ) : null}

      {adminToolsLoading && adminTools.length === 0 ? (
        <div className="text-center py-4">
          <span className="spinner-border text-secondary" />
        </div>
      ) : null}

      {!adminToolsLoading && adminTools.length === 0 ? (
        <p className="text-muted">Aucun outil.</p>
      ) : null}

      {adminTools.length > 0 ? (
        <div className="table-responsive">
          <table className="table table-dark table-hover table-sm align-middle">
            <thead>
              <tr>
                <th>Name</th>
                <th>Slug</th>
                <th>Version</th>
                <th>Status</th>
                <th>Created At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {adminTools.map((tool) => (
                <>
                  <tr
                    key={tool.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setExpandedToolId((prev) => (prev === tool.id ? null : tool.id))}
                  >
                    <td>{tool.name}</td>
                    <td>
                      <code className="small">{tool.slug}</code>
                    </td>
                    <td>{tool.version}</td>
                    <td>{toolStatusBadge(tool.status)}</td>
                    <td>{tool.created_at ? formatDate(tool.created_at) : '—'}</td>
                    <td onClick={(e) => e.stopPropagation()}>
                      {tool.status !== 'disabled' && tool.status !== 'archived' ? (
                        <button
                          className="btn btn-sm btn-warning me-1"
                          onClick={() => void patchToolStatus(tool.id, 'disabled')}
                          disabled={toolActionLoading !== null}
                        >
                          Disable
                        </button>
                      ) : null}
                      {tool.status !== 'archived' ? (
                        <button
                          className="btn btn-sm btn-danger"
                          onClick={() => void patchToolStatus(tool.id, 'archived')}
                          disabled={toolActionLoading !== null}
                        >
                          Archive
                        </button>
                      ) : null}
                    </td>
                  </tr>
                  {expandedToolId === tool.id ? (
                    <tr key={`${tool.id}-detail`}>
                      <td colSpan={6}>
                        <div className="p-2 small">
                          {tool.description ? (
                            <p className="mb-1">
                              <strong>Description:</strong> {tool.description}
                            </p>
                          ) : null}
                          <p className="mb-1">
                            <strong>Test runs:</strong>{' '}
                            {tool.test_runs ? tool.test_runs.length : 0}
                          </p>
                          {tool.test_runs && tool.test_runs.length > 0 ? (
                            <table className="table table-dark table-sm mt-1 mb-0">
                              <thead>
                                <tr>
                                  <th>#</th>
                                  <th>Status</th>
                                  <th>Date</th>
                                </tr>
                              </thead>
                              <tbody>
                                {tool.test_runs.map((run) => (
                                  <tr key={run.id}>
                                    <td>{run.attempt_number}</td>
                                    <td>{toolStatusBadge(run.status)}</td>
                                    <td>{run.created_at ? formatDate(run.created_at) : '—'}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  ) : null}
                </>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );

  const renderAdminSkills = () => (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Skills</h5>
        <button
          className="btn btn-sm btn-outline-secondary"
          onClick={() => void fetchAdminSkills()}
          disabled={adminSkillsLoading}
        >
          {adminSkillsLoading ? (
            <span className="spinner-border spinner-border-sm me-1" />
          ) : null}
          Rafraichir
        </button>
      </div>

      {adminSkillsError ? (
        <div className="alert alert-danger">{adminSkillsError}</div>
      ) : null}

      {adminSkillsLoading && adminSkills.length === 0 ? (
        <div className="text-center py-4">
          <span className="spinner-border text-secondary" />
        </div>
      ) : null}

      {!adminSkillsLoading && adminSkills.length === 0 ? (
        <p className="text-muted">Aucune skill.</p>
      ) : null}

      {adminSkills.length > 0 ? (
        <div className="table-responsive">
          <table className="table table-dark table-hover table-sm align-middle">
            <thead>
              <tr>
                <th>Title</th>
                <th>Slug</th>
                <th>Status</th>
                <th>Summary</th>
                <th>Created At</th>
              </tr>
            </thead>
            <tbody>
              {adminSkills.map((skill) => (
                <>
                  <tr
                    key={skill.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setExpandedSkillId((prev) => (prev === skill.id ? null : skill.id))}
                  >
                    <td>{skill.title}</td>
                    <td>
                      <code className="small">{skill.slug}</code>
                    </td>
                    <td>{skillStatusBadge(skill.status)}</td>
                    <td>
                      <span title={skill.summary || ''}>
                        {skill.summary && skill.summary.length > 60
                          ? `${skill.summary.slice(0, 57)}...`
                          : (skill.summary ?? '—')}
                      </span>
                    </td>
                    <td>{skill.created_at ? formatDate(skill.created_at) : '—'}</td>
                  </tr>
                  {expandedSkillId === skill.id && skill.markdown_content ? (
                    <tr key={`${skill.id}-content`}>
                      <td colSpan={5}>
                        <div className="p-2">
                          <pre
                            style={{
                              maxHeight: '300px',
                              overflowY: 'auto',
                              background: '#0d0d1a',
                              padding: '0.75rem',
                              borderRadius: '4px',
                              fontSize: '0.8rem',
                            }}
                          >
                            {skill.markdown_content}
                          </pre>
                        </div>
                      </td>
                    </tr>
                  ) : null}
                </>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );

  const renderAdminLogs = () => (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Audit Logs</h5>
        <button
          className="btn btn-sm btn-outline-secondary"
          onClick={() => void fetchAuditLogs()}
          disabled={auditLogsLoading}
        >
          {auditLogsLoading ? (
            <span className="spinner-border spinner-border-sm me-1" />
          ) : null}
          Rafraichir
        </button>
      </div>

      <div className="mb-3">
        <input
          type="text"
          className="form-control form-control-sm"
          placeholder="Filtrer par action..."
          value={logsActionFilter}
          onChange={(e) => setLogsActionFilter(e.target.value)}
          style={{ maxWidth: '300px', background: '#1e1e2e', color: '#cdd6f4', border: '1px solid #45475a' }}
        />
      </div>

      {auditLogsError ? (
        <div className="alert alert-danger">{auditLogsError}</div>
      ) : null}

      {auditLogsLoading && auditLogs.length === 0 ? (
        <div className="text-center py-4">
          <span className="spinner-border text-secondary" />
        </div>
      ) : null}

      {!auditLogsLoading && filteredAuditLogs.length === 0 ? (
        <p className="text-muted">Aucun log.</p>
      ) : null}

      {filteredAuditLogs.length > 0 ? (
        <div className="table-responsive">
          <table className="table table-dark table-hover table-sm align-middle">
            <thead>
              <tr>
                <th>Date</th>
                <th>Actor</th>
                <th>Action</th>
                <th>Target Type</th>
                <th>Target ID</th>
                <th>Metadata</th>
              </tr>
            </thead>
            <tbody>
              {filteredAuditLogs.map((log) => (
                <tr key={log.id}>
                  <td>{log.created_at ? formatDate(log.created_at) : '—'}</td>
                  <td>{log.actor_type}</td>
                  <td>
                    <code className="small">{log.action}</code>
                  </td>
                  <td>{log.target_type ?? '—'}</td>
                  <td>
                    {log.target_id ? (
                      <code className="small">{log.target_id.slice(0, 8)}</code>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td>
                    {log.metadata ? (
                      <span
                        title={JSON.stringify(log.metadata)}
                        style={{ cursor: 'help', textDecoration: 'underline dotted' }}
                      >
                        {'{...}'}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );

  const renderAdminRegistry = () => (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Registry — Pending Approval</h5>
        <button
          className="btn btn-sm btn-outline-secondary"
          onClick={() => void fetchRegistryMissions()}
          disabled={registryLoading}
        >
          {registryLoading ? (
            <span className="spinner-border spinner-border-sm me-1" />
          ) : null}
          Rafraichir
        </button>
      </div>

      {registryError ? (
        <div className="alert alert-danger">{registryError}</div>
      ) : null}

      {registryLoading && registryMissions.length === 0 ? (
        <div className="text-center py-4">
          <span className="spinner-border text-secondary" />
        </div>
      ) : null}

      {!registryLoading && registryMissions.length === 0 ? (
        <p className="text-muted">Aucune mission en attente d'approbation.</p>
      ) : null}

      {registryMissions.length > 0 ? (
        <div className="table-responsive">
          <table className="table table-dark table-hover table-sm align-middle">
            <thead>
              <tr>
                <th>Mission ID</th>
                <th>Prompt</th>
                <th>Autonomy</th>
                <th>Validation Status</th>
                <th>Started At</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {registryMissions.map((m) => (
                <tr key={m.id}>
                  <td>
                    <code className="small">{m.id.slice(0, 8)}</code>
                  </td>
                  <td>
                    <span title={m.prompt}>
                      {m.prompt.length > 60 ? `${m.prompt.slice(0, 57)}...` : m.prompt}
                    </span>
                  </td>
                  <td>{m.autonomy_level}</td>
                  <td>{validationStatusBadge('pending')}</td>
                  <td>{m.started_at ? formatDate(m.started_at) : '—'}</td>
                  <td>
                    <button
                      className="btn btn-sm btn-success me-1"
                      onClick={() => void approveRegistryMission(m.id)}
                      disabled={registryActionLoading !== null}
                    >
                      {registryActionLoading === m.id + ':approve' ? (
                        <span className="spinner-border spinner-border-sm me-1" />
                      ) : null}
                      Approve
                    </button>
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => void rejectRegistryMission(m.id)}
                      disabled={registryActionLoading !== null}
                    >
                      {registryActionLoading === m.id + ':reject' ? (
                        <span className="spinner-border spinner-border-sm me-1" />
                      ) : null}
                      Reject
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );

  // ─── JSX ────────────────────────────────────────────────────────────────────

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* T32 — Top Navigation Bar */}
      <nav
        style={{
          background: '#1e1e2e',
          borderBottom: '1px solid #313244',
          padding: '0 1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0',
          flexShrink: 0,
          height: '48px',
          zIndex: 100,
        }}
      >
        <span
          style={{
            color: '#cba6f7',
            fontWeight: 700,
            fontSize: '1rem',
            marginRight: '2rem',
            whiteSpace: 'nowrap',
          }}
        >
          The Living Kernel
        </span>

        {(['chat', 'missions', 'admin'] as ViewType[]).map((view) => (
          <button
            key={view}
            type="button"
            onClick={() => setCurrentView(view)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: currentView === view ? '2px solid #cba6f7' : '2px solid transparent',
              color: currentView === view ? '#cba6f7' : '#a6adc8',
              padding: '0 1rem',
              height: '48px',
              cursor: 'pointer',
              fontWeight: currentView === view ? 600 : 400,
              fontSize: '0.9rem',
              textTransform: 'capitalize',
              transition: 'color 0.15s',
            }}
          >
            {view === 'chat' ? 'Chat' : view === 'missions' ? 'Missions' : 'Admin'}
          </button>
        ))}

        <button
          key="documents-tab"
          type="button"
          onClick={() => setCurrentView('chat')}
          style={{
            background: 'none',
            border: 'none',
            borderBottom: '2px solid transparent',
            color: '#a6adc8',
            padding: '0 1rem',
            height: '48px',
            cursor: 'pointer',
            fontWeight: 400,
            fontSize: '0.9rem',
            transition: 'color 0.15s',
          }}
        >
          Documents
        </button>
      </nav>

      {/* Chat View */}
      {currentView === 'chat' ? (
        <div className="chat-app-shell" style={{ flex: 1, overflow: 'hidden' }}>
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

            <section className="docs-panel">
              <div className="docs-header">
                <div>
                  <strong>Documents RAG</strong>
                  {latestDocumentDate ? <div className="docs-subtitle">Maj: {latestDocumentDate}</div> : null}
                </div>
                <span>{documents.length}</span>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                className="hidden-file-input"
                multiple
                accept=".pdf,.txt,.md"
                onChange={handleFilesSelected}
              />

              <div className="docs-actions">
                <button type="button" onClick={handleOpenFilePicker} disabled={isUploadingDocs}>
                  {isUploadingDocs ? 'Import...' : 'Importer'}
                </button>
                <button type="button" onClick={() => void fetchDocuments()} disabled={isDocumentsLoading}>
                  Rafraichir
                </button>
              </div>

              {documentsStatus ? <div className="docs-status">{documentsStatus}</div> : null}
              {documentsError ? <div className="docs-error">{documentsError}</div> : null}

              <div className="docs-list">
                {isDocumentsLoading ? <div className="docs-empty">Chargement...</div> : null}

                {!isDocumentsLoading && documents.length === 0 ? (
                  <div className="docs-empty">Aucun document indexe.</div>
                ) : null}

                {!isDocumentsLoading &&
                  documents.map((doc) => (
                    <article key={doc.doc_id} className="doc-item">
                      <div className="doc-title">{doc.filename}</div>
                      <div className="doc-meta">
                        <span>{doc.file_type?.toUpperCase() || 'DOC'}</span>
                        <span>{doc.chunks ?? 0} chunks</span>
                        <span>{formatBytes(doc.size_bytes)}</span>
                        <span>{formatDate(doc.created_at)}</span>
                      </div>
                      <button
                        type="button"
                        className="doc-delete"
                        onClick={() => void deleteDocument(doc.doc_id)}
                        disabled={isUploadingDocs || isDocumentsLoading}
                      >
                        Suppr.
                      </button>
                    </article>
                  ))}
              </div>
            </section>
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
                <label className={`rag-chip ${useRag ? 'enabled' : ''}`}>
                  <input
                    type="checkbox"
                    checked={useRag}
                    onChange={() => setUseRag((prev) => !prev)}
                    disabled={isLoading}
                  />
                  Mode RAG ({documents.length})
                </label>

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
                      {msg.role === 'assistant' && msg.usedRag ? (
                        <div className="rag-sources">
                          <div className="rag-sources-title">Sources</div>
                          {msg.sources && msg.sources.length > 0 ? (
                            <div className="rag-source-list">
                              {msg.sources.map((source, index) => (
                                <article
                                  key={source.source_id || `${msg.id}-source-${index}`}
                                  className="rag-source-item"
                                >
                                  <div className="rag-source-top">
                                    <span className="rag-source-name">
                                      {source.filename || source.doc_id || `Source ${index + 1}`}
                                    </span>
                                    <span className="rag-source-score">
                                      score: {formatRelevanceScore(source.score)}
                                    </span>
                                  </div>
                                  {typeof source.chunk_index === 'number' ? (
                                    <div className="rag-source-chunk">chunk #{source.chunk_index}</div>
                                  ) : null}
                                  {source.preview ? (
                                    <p className="rag-source-preview">{source.preview}</p>
                                  ) : null}
                                </article>
                              ))}
                            </div>
                          ) : (
                            <p className="rag-sources-empty">
                              Aucune source pertinente n a ete retournee pour cette reponse.
                            </p>
                          )}
                        </div>
                      ) : null}
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
                    <span className="typing-label">
                      {useRag ? 'Analyse des documents...' : 'Generation de la reponse...'}
                    </span>
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
      ) : null}

      {/* T27 — Missions View */}
      {currentView === 'missions' ? (
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', background: '#181825' }}>
          <div style={{ maxWidth: '960px', margin: '0 auto' }}>
            <div className="d-flex justify-content-between align-items-center mb-4">
              <h4 style={{ color: '#cba6f7', margin: 0 }}>Missions</h4>
              <button
                className="btn btn-sm btn-outline-secondary"
                onClick={() => void fetchMissions()}
                disabled={missionsLoading}
              >
                {missionsLoading ? (
                  <span className="spinner-border spinner-border-sm me-1" />
                ) : null}
                Rafraichir
              </button>
            </div>

            {/* Create mission form */}
            <div
              style={{
                background: '#1e1e2e',
                border: '1px solid #313244',
                borderRadius: '8px',
                padding: '1rem',
                marginBottom: '1.5rem',
              }}
            >
              <h6 style={{ color: '#cdd6f4', marginBottom: '0.75rem' }}>Nouvelle mission</h6>
              <textarea
                className="form-control mb-2"
                rows={3}
                placeholder="Decris la mission a accomplir..."
                value={newMissionPrompt}
                onChange={(e) => setNewMissionPrompt(e.target.value)}
                style={{ background: '#181825', color: '#cdd6f4', border: '1px solid #45475a' }}
              />
              <div className="d-flex gap-2 align-items-center">
                <select
                  className="form-select form-select-sm"
                  value={newMissionAutonomy}
                  onChange={(e) => setNewMissionAutonomy(e.target.value)}
                  style={{ background: '#181825', color: '#cdd6f4', border: '1px solid #45475a', maxWidth: '180px' }}
                >
                  <option value="restricted">restricted</option>
                  <option value="supervised">supervised</option>
                  <option value="extended">extended</option>
                </select>
                <button
                  className="btn btn-sm btn-primary"
                  onClick={() => void createMission()}
                  disabled={missionCreateLoading || !newMissionPrompt.trim()}
                >
                  {missionCreateLoading ? (
                    <span className="spinner-border spinner-border-sm me-1" />
                  ) : null}
                  Creer la mission
                </button>
              </div>
              {missionCreateError ? (
                <div className="alert alert-danger mt-2 mb-0 py-1 px-2 small">{missionCreateError}</div>
              ) : null}
            </div>

            {missionsError ? (
              <div className="alert alert-danger">{missionsError}</div>
            ) : null}

            {missionsLoading && missions.length === 0 ? (
              <div className="text-center py-5">
                <span className="spinner-border text-secondary" />
              </div>
            ) : null}

            {!missionsLoading && missions.length === 0 ? (
              <p style={{ color: '#6c7086' }}>Aucune mission pour l instant.</p>
            ) : null}

            {missions.length > 0 ? (
              <div className="table-responsive">
                <table className="table table-dark table-hover align-middle">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Prompt</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {missions.map((m) => (
                      <>
                        <tr
                          key={m.id}
                          style={{ cursor: 'pointer' }}
                          onClick={() => setExpandedMissionId((prev) => (prev === m.id ? null : m.id))}
                        >
                          <td style={{ whiteSpace: 'nowrap' }}>
                            {m.started_at ? formatDate(m.started_at) : '—'}
                          </td>
                          <td>
                            <span title={m.prompt}>
                              {m.prompt.length > 80 ? `${m.prompt.slice(0, 77)}...` : m.prompt}
                            </span>
                          </td>
                          <td>{missionStatusBadge(m.status)}</td>
                          <td onClick={(e) => e.stopPropagation()}>
                            <div className="d-flex gap-1 flex-wrap">
                              {m.status === 'pending' ? (
                                <button
                                  className="btn btn-sm btn-primary"
                                  onClick={() => void runMission(m.id)}
                                  disabled={missionActionLoading !== null}
                                >
                                  {missionActionLoading === m.id + ':run' ? (
                                    <span className="spinner-border spinner-border-sm me-1" />
                                  ) : null}
                                  Run
                                </button>
                              ) : null}
                              {m.status === 'awaiting_approval' ? (
                                <>
                                  <button
                                    className="btn btn-sm btn-success"
                                    onClick={() => void approveMission(m.id)}
                                    disabled={missionActionLoading !== null}
                                  >
                                    {missionActionLoading === m.id + ':approve' ? (
                                      <span className="spinner-border spinner-border-sm me-1" />
                                    ) : null}
                                    Approve
                                  </button>
                                  <button
                                    className="btn btn-sm btn-danger"
                                    onClick={() => void rejectMission(m.id)}
                                    disabled={missionActionLoading !== null}
                                  >
                                    {missionActionLoading === m.id + ':reject' ? (
                                      <span className="spinner-border spinner-border-sm me-1" />
                                    ) : null}
                                    Reject
                                  </button>
                                </>
                              ) : null}
                            </div>
                          </td>
                        </tr>
                        {expandedMissionId === m.id ? (
                          <tr key={`${m.id}-expand`}>
                            <td colSpan={4}>
                              <div style={{ padding: '0.5rem 1rem' }}>
                                {m.steps && m.steps.length > 0 ? (
                                  <table className="table table-dark table-sm mb-0">
                                    <thead>
                                      <tr>
                                        <th>Node</th>
                                        <th>Status</th>
                                        <th>Started At</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {m.steps.map((step) => (
                                        <tr key={step.id}>
                                          <td>{step.node_name}</td>
                                          <td>{missionStatusBadge(step.status)}</td>
                                          <td>{step.started_at ? formatDate(step.started_at) : '—'}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                ) : (
                                  <p className="small text-muted mb-0">Aucun step disponible.</p>
                                )}
                              </div>
                            </td>
                          </tr>
                        ) : null}
                      </>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Admin View */}
      {currentView === 'admin' ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#181825' }}>
          {/* Admin sub-nav */}
          <div
            style={{
              background: '#1e1e2e',
              borderBottom: '1px solid #313244',
              padding: '0 1.5rem',
              display: 'flex',
              gap: '0',
              flexShrink: 0,
            }}
          >
            {(['missions', 'tools', 'skills', 'logs', 'registry'] as AdminTabType[]).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setAdminTab(tab)}
                style={{
                  background: 'none',
                  border: 'none',
                  borderBottom: adminTab === tab ? '2px solid #89b4fa' : '2px solid transparent',
                  color: adminTab === tab ? '#89b4fa' : '#a6adc8',
                  padding: '0.6rem 1rem',
                  cursor: 'pointer',
                  fontWeight: adminTab === tab ? 600 : 400,
                  fontSize: '0.85rem',
                  textTransform: 'capitalize',
                  transition: 'color 0.15s',
                }}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Admin tab content */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem' }}>
            <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
              {adminTab === 'missions' ? renderAdminMissions() : null}
              {adminTab === 'tools' ? renderAdminTools() : null}
              {adminTab === 'skills' ? renderAdminSkills() : null}
              {adminTab === 'logs' ? renderAdminLogs() : null}
              {adminTab === 'registry' ? renderAdminRegistry() : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;

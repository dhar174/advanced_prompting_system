export interface ConversationInputType {
  role: string;
  name?: string | null;
  content: string;
}

export interface FeedbackInputType {
  assistant: string;
  rating: number;
}

export interface QuestionInputType {
  assistant: string;
  question: string;
}

export interface ConversationType {
  role: string;
  name?: string | null;
  content: string;
}

export interface QuestionType {
  assistant: string;
  question: string;
}

// ============================================================================
// Execution Modes & Observability Types (Issue #35 Phases 3–5 & Issue #21)
// ============================================================================

export type ExecutionModeId = 'collaborative' | 'advanced_pipeline' | 'tree_of_thought';

export interface ExecutionModeCapabilityType {
  id: ExecutionModeId;
  label: string;
  description: string;
  enabled: boolean;
  gatedIssue?: string | null;
  supportsMultiAgent: boolean;
  supportsStepPlanning: boolean;
}

export interface OutputTypeInfo {
  outputType: string;
  fileExtension: string;
}

export interface SubtaskDiagnosticType {
  subtaskNumber: number;
  subtaskName: string;
  subtaskDescription: string;
  subtaskOutput: string;
  completed: boolean;
}

export interface PlanStepDiagnosticType {
  stepNumber: number;
  stepName: string;
  stepDescription: string;
  stepExplanation: string;
  stepOutput: string;
  completed: boolean;
  assignedAgents: string[];
  subtasks: SubtaskDiagnosticType[];
}

export interface ReflectionDiagnosticType {
  stepNumber: number;
  planStepNumber?: number | null;
  stepDescription: string;
  thoughts?: string | null;
  content: string;
  reward: number;
  remainingBudget: number;
}

export interface AgentAttributionType {
  agentName: string;
  role: string;
  roundOrStep: number;
  contributionSummary: string;
}

export interface RunDiagnosticsType {
  complexityScore?: number | null;
  initialBudget?: number | null;
  adjustedBudget?: number | null;
  finalReward?: number | null;
  durationMs?: number | null;
  tokenCount?: number | null;
  correlationId?: string | null;
  planSteps: PlanStepDiagnosticType[];
  reflections: ReflectionDiagnosticType[];
  agentAttributions: AgentAttributionType[];
}

export interface RunEventType {
  eventId: string;
  eventType: string;
  timestamp: string;
  mode: string;
  stepNumber?: number | null;
  planStepNumber?: number | null;
  agentName?: string | null;
  summary: string;
}

export interface RunExecutionInputType {
  mode: string;
  taskPrompt?: string | null;
  conversation?: ConversationInputType[] | null;
  assistantPersonalities?: string[] | null;
  leadPersonality?: string | null;
  numRounds?: number;
  maxSteps?: number;
  initialBudget?: number;
  agents?: number;
  model?: string;
  temperature?: number;
  context?: string | null;
}

export interface RunExecutionResponseType {
  mode: ExecutionModeId;
  status: 'completed' | 'gated' | 'error';
  conversation: ConversationType[];
  questions?: QuestionType[] | null;
  finalOutput: string;
  outputType: OutputTypeInfo;
  diagnostics: RunDiagnosticsType;
  events: RunEventType[];
  errorMessage?: string | null;
}

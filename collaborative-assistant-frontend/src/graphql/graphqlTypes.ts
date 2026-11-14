export interface ConversationInputType {
  role: string;
  name?: string | null;
  content: string;
}

export interface FeedbackInputType {
  assistant: string;
  rating: number;
  comments?: string | null;
}

export interface QuestionInputType {
  assistant: string;
  question: string;
}

export interface ConversationType {
  planStepNumber?: number | null;
  role: string;
  name?: string | null;
  content: string;
}

export interface QuestionType {
  planStepNumber?: number | null;
  assistant: string;
  question: string;
}

// ============================================================================
// Execution Modes & Observability Types (Issue #35 Phases 3–5 & Issue #21)
// ============================================================================

export type ExecutionModeId = 'collaborative' | 'advanced_pipeline' | 'tree_of_thought';

export type RunStatus =
  | 'completed'
  | 'gated'
  | 'error'
  | 'incomplete'
  | 'budget_exhausted'
  | 'cancelled'
  | 'timeout';

export interface ExecutionModeCapabilityType {
  id: ExecutionModeId;
  label: string;
  description: string;
  enabled: boolean;
  gatedIssue?: string | null;
  supportsMultiAgent: boolean;
  supportsStepPlanning: boolean;
  ready?: boolean;
  unavailableReason?: string | null;
  supportsCancellation?: boolean;
  supportsResume?: boolean;
  minNumRounds?: number | null;
  maxNumRounds?: number | null;
  maxSteps?: number | null;
  maxAgents?: number | null;
}

export interface PersonalityCapabilityType {
  id: string;
  instructions: string;
}

export interface ProviderPolicyType {
  defaultModel: string;
  allowedModels: string[];
  defaultTemperature: number;
}

export interface ArtifactRefType {
  artifactId: string;
  runId: string;
  filename: string;
  path: string;
  mimeType: string;
  sizeBytes: number;
  sha256: string;
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
  model?: string | null;
  temperature?: number;
  context?: string | null;
}

export interface RunExecutionResponseType {
  mode: ExecutionModeId;
  effectiveMode?: string | null;
  status: RunStatus;
  stopReason?: string | null;
  conversation: ConversationType[];
  questions?: QuestionType[] | null;
  finalOutput: string;
  outputType: OutputTypeInfo;
  diagnostics: RunDiagnosticsType;
  events: RunEventType[];
  artifacts?: ArtifactRefType[];
  usageAvailable?: boolean;
  errorMessage?: string | null;
}

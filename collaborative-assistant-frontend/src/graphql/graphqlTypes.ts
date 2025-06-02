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

export interface SubtaskType {
  subtask_number: number;
  completed: boolean;
  subtask_description: string;
  subtask_name: string;
  subtask_explanation: string;
  subtask_output: string;
  subtask_full_text: string;
  subtasks: SubtaskType[];
}

export interface PlanStepType {
  step_number: number;
  completed: boolean;
  step_name: string;
  step_description: string;
  step_explanation: string;
  step_output: string;
  step_full_text: string;
  subtasks: SubtaskType[];
}

export interface PlanType {
  steps: PlanStepType[];
}

export interface ConversationMemoryType {
  facts: string[];
  arguments: string[];
  decisions: string[];
  direct_replies: string[];
  recommended_actions: string[];
  to_do_list: string[];
  completed_tasks: string[];
  rounds_left: number;
  decided_output_type?: string | null;
}

export interface AgentCollaborationType {
  agent_name: string;
  priority_score: number;
  contributions: string[];
  votes_cast: string[];
  questions_asked: string[];
}

export interface ComplexityMetricsType {
  overall_score: number;
  reasoning_depth: number;
  solution_complexity: number;
  collaboration_intensity: number;
  confidence_level: number;
}

export interface ProcessingStatusType {
  current_round: number;
  total_rounds: number;
  current_step: string;
  progress_percentage: number;
  estimated_time_remaining?: number | null;
}

export interface RunConversationResponseType {
  conversation: ConversationType[];
  questions?: QuestionType[] | null;
  final_output: string;
  plan?: PlanType | null;
  conversation_memory?: ConversationMemoryType | null;
  agent_collaboration: AgentCollaborationType[];
  complexity_metrics?: ComplexityMetricsType | null;
  processing_status?: ProcessingStatusType | null;
}

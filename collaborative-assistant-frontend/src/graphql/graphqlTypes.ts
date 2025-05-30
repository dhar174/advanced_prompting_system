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

import { gql } from '@apollo/client';
import type {
  ExecutionModeCapabilityType,
  PersonalityCapabilityType,
  ProviderPolicyType,
  RunExecutionInputType,
  RunExecutionResponseType,
} from './graphqlTypes';

export const GET_EXECUTION_MODES = gql`
  query GetExecutionModes {
    executionModes {
      id
      label
      description
      enabled
      gatedIssue
      supportsMultiAgent
      supportsStepPlanning
      ready
      unavailableReason
      supportsCancellation
      supportsResume
    }
  }
`;

export interface GetExecutionModesData {
  executionModes: ExecutionModeCapabilityType[];
}

export const GET_PERSONALITIES = gql`
  query GetPersonalities {
    personalities {
      id
      instructions
    }
  }
`;

export interface GetPersonalitiesData {
  personalities: PersonalityCapabilityType[];
}

export const GET_PROVIDER_POLICY = gql`
  query GetProviderPolicy {
    providerPolicy {
      defaultModel
      allowedModels
      defaultTemperature
    }
  }
`;

export interface GetProviderPolicyData {
  providerPolicy: ProviderPolicyType;
}

export const RUN_EXECUTION = gql`
  mutation RunExecution($input: RunExecutionInput!) {
    runExecution(input: $input) {
      mode
      effectiveMode
      status
      stopReason
      finalOutput
      errorMessage
      usageAvailable
      outputType {
        outputType
        fileExtension
      }
      conversation {
        role
        name
        content
      }
      questions {
        assistant
        question
      }
      artifacts {
        artifactId
        runId
        filename
        path
        mimeType
        sizeBytes
        sha256
      }
      diagnostics {
        complexityScore
        initialBudget
        adjustedBudget
        finalReward
        durationMs
        tokenCount
        correlationId
        planSteps {
          stepNumber
          stepName
          stepDescription
          stepExplanation
          stepOutput
          completed
          assignedAgents
          subtasks {
            subtaskNumber
            subtaskName
            subtaskDescription
            subtaskOutput
            completed
          }
        }
        reflections {
          stepNumber
          planStepNumber
          stepDescription
          thoughts
          content
          reward
          remainingBudget
        }
        agentAttributions {
          agentName
          role
          roundOrStep
          contributionSummary
        }
      }
      events {
        eventId
        eventType
        timestamp
        mode
        stepNumber
        planStepNumber
        agentName
        summary
      }
    }
  }
`;

export interface RunExecutionVariables {
  input: RunExecutionInputType;
}

export interface RunExecutionData {
  runExecution: RunExecutionResponseType;
}

export const RUN_ADVANCED_PIPELINE = gql`
  mutation RunAdvancedPipeline(
    $taskPrompt: String!
    $maxSteps: Int
    $initialBudget: Int
    $agents: Int
    $model: String
  ) {
    runAdvancedPipeline(
      taskPrompt: $taskPrompt
      maxSteps: $maxSteps
      initialBudget: $initialBudget
      agents: $agents
      model: $model
    ) {
      mode
      effectiveMode
      status
      stopReason
      finalOutput
      errorMessage
      usageAvailable
      outputType {
        outputType
        fileExtension
      }
      conversation {
        role
        name
        content
      }
      artifacts {
        artifactId
        runId
        filename
        path
        mimeType
        sizeBytes
        sha256
      }
      diagnostics {
        complexityScore
        initialBudget
        adjustedBudget
        finalReward
        durationMs
        tokenCount
        correlationId
        planSteps {
          stepNumber
          stepName
          stepDescription
          stepExplanation
          stepOutput
          completed
          assignedAgents
          subtasks {
            subtaskNumber
            subtaskName
            subtaskDescription
            subtaskOutput
            completed
          }
        }
        reflections {
          stepNumber
          planStepNumber
          stepDescription
          thoughts
          content
          reward
          remainingBudget
        }
        agentAttributions {
          agentName
          role
          roundOrStep
          contributionSummary
        }
      }
      events {
        eventId
        eventType
        timestamp
        mode
        stepNumber
        planStepNumber
        agentName
        summary
      }
    }
  }
`;

export interface RunAdvancedPipelineVariables {
  taskPrompt: string;
  maxSteps?: number;
  initialBudget?: number;
  agents?: number;
  model?: string;
}

export interface RunAdvancedPipelineData {
  runAdvancedPipeline: RunExecutionResponseType;
}

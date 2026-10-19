import type { ChatResponse, Message } from '@/types'

export interface LocalMessage extends Omit<Message, 'role'> {
  role: 'user' | 'assistant'
  domain?: string
  confidence?: number | null
  reason?: string | null
  sources?: string[]
  warning?: string
  explainability?: Record<string, unknown>
}

function cloneExplainability(
  explainability?: Record<string, unknown> | null
): Record<string, unknown> | undefined {
  return explainability
    ? JSON.parse(JSON.stringify(explainability)) as Record<string, unknown>
    : undefined
}

export function mapConversationMessages(
  messageList: Message[],
  previousMessages: LocalMessage[],
  response?: ChatResponse
): LocalMessage[] {
  const latestAssistantId = response?.messages
    ?.slice()
    .reverse()
    .find((message) => message.role === 'assistant')?.id
    ?? (messageList
      .slice()
      .reverse()
      .find((message) => message.role === 'assistant')?.id)

  return messageList.map((message) => {
    const existingMessage = previousMessages.find((item) => item.id === message.id)
    const isCurrentResponse = message.id === latestAssistantId && message.role === 'assistant'

    return {
      ...message,
      role: message.role as 'user' | 'assistant',
      domain: existingMessage?.domain ?? (isCurrentResponse ? response?.domain : undefined),
      confidence: existingMessage?.confidence ?? (isCurrentResponse ? response?.confidence : undefined),
      reason: existingMessage?.reason ?? (isCurrentResponse ? response?.reason : undefined),
      sources: existingMessage?.sources
        ?? (isCurrentResponse ? [...(response?.sources ?? [])] : undefined),
      warning: existingMessage?.warning,
      explainability: existingMessage?.explainability
        ?? (isCurrentResponse ? cloneExplainability(response?.explainability) : undefined),
    }
  })
}

export function cloneResponseExplainability(
  explainability?: Record<string, unknown> | null
): Record<string, unknown> | undefined {
  return cloneExplainability(explainability)
}

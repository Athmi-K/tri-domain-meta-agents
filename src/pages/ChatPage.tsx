import { useMemo, useState, useRef, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Send, Plus, Sparkles } from 'lucide-react'
import { toast } from 'sonner'
import { useProfile, useChatHistory, useConversation, useSendChat } from '@/hooks'
import { getErrorMessage } from '@/services'
import { PageHeader } from '@/components/layout/PageHeader'
import { AIResponseCard } from '@/components/common/AIResponseCard'
import { ConversationCard } from '@/components/common/ConversationCard'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { SUGGESTED_PROMPTS } from '@/utils/constants'
import type { ChatResponse } from '@/types'

interface LocalMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  domain?: string
  confidence?: number | null
  reason?: string | null
  sources?: string[]
  warning?: string
  explainability?: Record<string, unknown>
}

export function ChatPage() {
  const location = useLocation()
  const scrollRef = useRef<HTMLDivElement>(null)

  const { data: profile } = useProfile()
  const suggestedPrompts = useMemo(() => {
    const prompts = [] as Array<{ domain: string; text: string }>

    if (profile?.career?.target_role) {
      prompts.push({
        domain: 'career',
        text: `What should I do next to become a ${profile.career.target_role}?`,
      })
    }

    if (profile?.health?.fitness_goal) {
      prompts.push({
        domain: 'health',
        text: `How can I improve my ${profile.health.fitness_goal} plan?`,
      })
    }

    if (profile?.finance?.monthly_income || profile?.finance?.monthly_expenses) {
      prompts.push({
        domain: 'finance',
        text: 'Help me optimize my budget and savings based on my current finances.',
      })
    }

    if (prompts.length === 0) {
      return SUGGESTED_PROMPTS.slice(0, 4)
    }

    return [...prompts, ...SUGGESTED_PROMPTS].slice(0, 4)
  }, [profile])

  const [messages, setMessages] = useState<LocalMessage[]>([])
  const [input, setInput] = useState('')
  const [domain, setDomain] = useState<string>('auto')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  const { data: history } = useChatHistory()
  const { data: conversation } = useConversation(conversationId)
  const sendChat = useSendChat()

  const conversations = history ?? []

  useEffect(() => {
    const state = location.state as { query?: string; conversationId?: string } | null
    if (state?.query) {
      setInput(state.query)
      window.history.replaceState({}, '')
    }
    if (state?.conversationId) {
      setConversationId(state.conversationId)
      window.history.replaceState({}, '')
    }
  }, [location.state])

  useEffect(() => {
    if (conversation?.messages) {
      setMessages((prev) =>
        conversation.messages.map((m) => {
          const existingMessage = prev.find((msg) => msg.id === m.id)

          return {
            id: m.id,
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: m.timestamp,
            domain: existingMessage?.domain ?? conversation.domain,
            confidence: existingMessage?.confidence ?? null,
            reason: existingMessage?.reason ?? null,
            sources: existingMessage?.sources,
            warning: existingMessage?.warning,
            explainability: existingMessage?.explainability,
          }
        })
      )
    }
  }, [conversation])

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isStreaming])

  const handleSend = async (text?: string) => {
    const query = (text || input).trim()
    if (!query || isStreaming) return

    const userMsg: LocalMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsStreaming(true)
    try {
      const res: ChatResponse = await sendChat.mutateAsync({
        query,
        domain: domain === 'auto' ? 'auto' : domain,
        conversation_id: conversationId,
      })
      if (!conversationId) setConversationId(res.conversation_id)

      if (res.messages?.length) {
        setMessages(
          res.messages.map((m) => ({
            id: m.id,
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: m.timestamp,
            domain: m.role === 'assistant' ? res.domain : undefined,
            confidence: m.role === 'assistant' ? res.confidence : undefined,
            reason: m.role === 'assistant' ? res.reason : undefined,
            sources: m.role === 'assistant' ? res.sources : undefined,
            explainability: m.role === 'assistant' ? res.explainability ?? undefined : undefined,
          }))
        )
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: res.answer,
            timestamp: new Date().toISOString(),
            domain: res.domain,
            confidence: res.confidence,
            reason: res.reason,
            sources: res.sources,
            explainability: res.explainability ?? undefined,
          },
        ])
      }
    } catch (err) {
      toast.error(getErrorMessage(err))
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request. Please try again.',
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setIsStreaming(false)
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setConversationId(null)
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] gap-4">
      <div className="hidden lg:flex w-72 flex-col rounded-xl border bg-card/50 backdrop-blur-sm">
        <div className="flex items-center justify-between p-4 border-b">
          <h3 className="text-sm font-semibold">Conversations</h3>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleNewChat}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>
        <ScrollArea className="flex-1 p-3 space-y-2">
          {conversations.map((conv) => (
            <ConversationCard
              key={conv.id}
              conversation={conv}
              isActive={conv.id === conversationId}
              onClick={() => setConversationId(conv.id)}
            />
          ))}
        </ScrollArea>
      </div>

      <div className="flex flex-1 flex-col rounded-xl border bg-card/30 backdrop-blur-sm overflow-hidden">
        <div className="flex items-center justify-between border-b px-4 py-3 gap-4 flex-wrap">
          <PageHeader title="Ask AI" description="Multi-domain advisory assistant" className="!flex-row !gap-2" />

          <div className="flex items-center gap-4 flex-wrap">
            <Tabs value={domain} onValueChange={setDomain} aria-label="Choose advisory domain">
              <TabsList className="h-8">
                <TabsTrigger value="auto" className="text-xs px-2">Auto</TabsTrigger>
                <TabsTrigger value="career" className="text-xs px-2">Career</TabsTrigger>
                <TabsTrigger value="health" className="text-xs px-2">Health</TabsTrigger>
                <TabsTrigger value="finance" className="text-xs px-2">Finance</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>

        <ScrollArea className="flex-1 p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full py-12">
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-center max-w-lg"
              >
                <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 shadow-glow">
                  <Sparkles className="h-8 w-8 text-white" />
                </div>
                <h2 className="text-xl font-bold mb-2">How can I help you today?</h2>
                <p className="text-muted-foreground mb-8">
                  Ask about career growth, health goals, or financial planning
                </p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {suggestedPrompts.map((prompt) => (
                    <Button
                      key={prompt.text}
                      variant="outline"
                      className="h-auto py-3 px-4 text-left justify-start text-xs whitespace-normal"
                      onClick={() => handleSend(prompt.text)}
                    >
                      {prompt.text}
                    </Button>
                  ))}
                </div>
              </motion.div>
            </div>
          ) : (
            <div className="space-y-6 max-w-3xl mx-auto">
              {messages.map((msg) => (
                <AIResponseCard
                  key={msg.id}
                  content={msg.content}
                  role={msg.role}
                  timestamp={msg.timestamp}
                  domain={msg.domain}
                  confidence={msg.confidence}
                  reason={msg.reason}
                  sources={msg.sources}
                  warning={msg.warning}
                  explainability={msg.explainability}
                  onRegenerate={msg.role === 'assistant' ? () => {
                    const lastUser = [...messages].reverse().find((m) => m.role === 'user')
                    if (lastUser) handleSend(lastUser.content)
                  } : undefined}
                />
              ))}
              {isStreaming && (
                <AIResponseCard content="" role="assistant" isStreaming />
              )}
              <div ref={scrollRef} />
            </div>
          )}
        </ScrollArea>

        <div className="border-t p-4">
          <div className="flex gap-2 max-w-3xl mx-auto">
            <Textarea
              placeholder="Ask about career, health, or finance..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              className="min-h-[44px] max-h-32 resize-none"
              rows={1}
            />
            <Button
              variant="gradient"
              size="icon"
              className="shrink-0 h-11 w-11"
              onClick={() => handleSend()}
              disabled={!input.trim() || isStreaming}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

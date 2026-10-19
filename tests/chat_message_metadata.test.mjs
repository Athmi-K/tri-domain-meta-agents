import assert from 'node:assert/strict'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import test from 'node:test'
import vm from 'node:vm'

const require = createRequire(import.meta.url)
const typescript = require('../node_modules/typescript')
const source = fs.readFileSync(
  new URL('../src/utils/chatMessageMetadata.ts', import.meta.url),
  'utf8'
)
const output = typescript.transpileModule(source, {
  compilerOptions: {
    module: typescript.ModuleKind.CommonJS,
    target: typescript.ScriptTarget.ES2020,
  },
}).outputText
const module = { exports: {} }
vm.runInNewContext(
  `(function (exports, module) { ${output} })(module.exports, module)`,
  { module, JSON }
)
const { mapConversationMessages } = module.exports

const message = (id, role, content) => ({
  id,
  role,
  content,
  timestamp: `${id}:00`,
})

test('preserves Career metadata when Finance response is merged', () => {
  const career = {
    ...message('assistant-career', 'assistant', 'Career answer'),
    domain: 'career',
    confidence: 0.8,
    explainability: { selected_agent: 'career' },
    sources: ['career-kb'],
  }
  const response = {
    conversation_id: 'conversation-1',
    domain: 'finance',
    answer: 'Finance answer',
    confidence: 0.7,
    sources: ['finance-kb'],
    explainability: { selected_agent: 'finance' },
    messages: [
      message('assistant-career', 'assistant', 'Career answer'),
      message('assistant-finance', 'assistant', 'Finance answer'),
    ],
  }

  const mapped = mapConversationMessages(response.messages, [career], response)

  assert.equal(mapped[0].domain, 'career')
  assert.equal(mapped[0].confidence, 0.8)
  assert.deepEqual(mapped[0].explainability, { selected_agent: 'career' })
  assert.equal(mapped[1].domain, 'finance')
  assert.equal(mapped[1].sources.length, 1)
  assert.equal(mapped[1].sources[0], 'finance-kb')
})

test('does not reuse explainability objects between messages', () => {
  const response = {
    conversation_id: 'conversation-1',
    domain: 'health',
    answer: 'Health answer',
    explainability: { selected_agent: 'health', nested: { value: 1 } },
    messages: [message('assistant-health', 'assistant', 'Health answer')],
  }
  const mapped = mapConversationMessages(response.messages, [], response)

  mapped[0].explainability.nested.value = 2
  assert.equal(response.explainability.nested.value, 1)
})

test('does not assign the latest domain to historical messages without metadata', () => {
  const response = {
    conversation_id: 'conversation-1',
    domain: 'finance',
    answer: 'Finance answer',
    messages: [
      message('assistant-old', 'assistant', 'Old answer'),
      message('assistant-new', 'assistant', 'New answer'),
    ],
  }

  const mapped = mapConversationMessages(response.messages, [], response)

  assert.equal(mapped[0].domain, undefined)
  assert.equal(mapped[1].domain, 'finance')
})

test('isolates metadata for Career, Health, and Finance sequence', () => {
  const responses = [
    ['career', 'career-answer', { selected_agent: 'career' }],
    ['health', 'health-answer', { selected_agent: 'health' }],
    ['finance', 'finance-answer', { selected_agent: 'finance' }],
  ]
  let stored = []

  for (const [domain, answer, explainability] of responses) {
    const assistantId = `${domain}-assistant`
    const response = {
      conversation_id: 'conversation-1',
      domain,
      answer,
      sources: [`${domain}-source`],
      explainability,
      messages: [
        ...stored.map((item) => message(item.id, item.role, item.content)),
        message(assistantId, 'assistant', answer),
      ],
    }
    stored = mapConversationMessages(response.messages, stored, response)
  }

  assert.deepEqual(
    stored
      .filter((item) => item.role === 'assistant')
      .map((item) => [item.domain, item.explainability.selected_agent]),
    [
      ['career', 'career'],
      ['health', 'health'],
      ['finance', 'finance'],
    ]
  )
})

test('isolates metadata for Finance followed by Career', () => {
  const finance = message('finance-assistant', 'assistant', 'Finance answer')
  const career = message('career-assistant', 'assistant', 'Career answer')
  const first = {
    conversation_id: 'conversation-1',
    domain: 'finance',
    answer: finance.content,
    sources: ['finance-source'],
    explainability: { selected_agent: 'finance' },
    messages: [finance],
  }
  const firstMapped = mapConversationMessages([finance], [], first)
  const second = {
    conversation_id: 'conversation-1',
    domain: 'career',
    answer: career.content,
    sources: ['career-source'],
    explainability: { selected_agent: 'career' },
    messages: [finance, career],
  }
  const secondMapped = mapConversationMessages([finance, career], firstMapped, second)

  assert.equal(secondMapped[0].domain, 'finance')
  assert.equal(secondMapped[1].domain, 'career')
  assert.equal(secondMapped[0].explainability.selected_agent, 'finance')
  assert.equal(secondMapped[1].explainability.selected_agent, 'career')
})

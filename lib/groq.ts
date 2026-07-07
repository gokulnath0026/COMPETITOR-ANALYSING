import OpenAI from 'openai'

export const GROQ_MODEL = 'openai/gpt-oss-120b'

// Built lazily (not at module load) so that Next.js's build-time page-data
// collection doesn't crash when GROQ_API_KEY isn't present in that environment.
let client: OpenAI | null = null
function getClient(): OpenAI {
  if (!client) {
    client = new OpenAI({
      apiKey: process.env.GROQ_API_KEY,
      baseURL: 'https://api.groq.com/openai/v1',
    })
  }
  return client
}

type ChatCompletionParams = OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming

// Groq's free tier has a low tokens-per-minute limit, so bursts of concurrent
// requests routinely hit 429s. Retry with backoff honoring the server's
// retry-after hint before giving up.
export async function createChatCompletion(
  params: Omit<ChatCompletionParams, 'model'> & { model?: string },
  maxRetries = 5
): Promise<OpenAI.Chat.Completions.ChatCompletion> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await getClient().chat.completions.create({
        model: GROQ_MODEL,
        ...params,
      })
    } catch (error: any) {
      const isRateLimit = error?.status === 429
      if (!isRateLimit || attempt >= maxRetries) {
        throw error
      }
      const retryAfter = parseFloat(error?.headers?.['retry-after'] ?? '')
      const waitSeconds = Number.isFinite(retryAfter) ? retryAfter : Math.pow(2, attempt)
      await new Promise((resolve) => setTimeout(resolve, (waitSeconds + 0.5) * 1000))
    }
  }
}

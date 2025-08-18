import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { text, context, learningMode, previousMessages } = await request.json()

    if (!text) {
      return NextResponse.json({ error: "Text is required" }, { status: 400 })
    }

    // Process text with spaCy (Python script)
    const spacyResponse = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).catch(() => null)

    let entities = []
    let spacyAnalysis = null

    if (spacyResponse?.ok) {
      spacyAnalysis = await spacyResponse.json()
      entities = spacyAnalysis.entities || []
    }

    const response = await generateResponseWithFalcon(text, spacyAnalysis, context, learningMode)
    const suggestions = generateSuggestions(text, spacyAnalysis, learningMode)
    const learningPoints = generateLearningPoints(text, spacyAnalysis, learningMode)
    const learningUpdate = calculateLearningUpdate(text, spacyAnalysis, learningMode)

    return NextResponse.json({
      response,
      entities,
      analysis: spacyAnalysis,
      suggestions,
      learningPoints,
      learningUpdate,
    })
  } catch (error) {
    console.error("Error processing request:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

async function generateResponseWithFalcon(
  text: string,
  analysis: any,
  context: string[] = [],
  learningMode = "casual",
): Promise<string> {
  try {
    // First, try Hugging Face API
    const hfResponse = await callHuggingFaceAPI(text, analysis, context, learningMode)
    if (hfResponse) {
      return hfResponse
    }

    // Fallback to local Falcon model
    const localResponse = await callLocalFalcon(text, analysis, context, learningMode)
    if (localResponse) {
      return localResponse
    }

    // Final fallback to enhanced rule-based responses
    return generateEnhancedFallbackResponse(text, analysis, context, learningMode)
  } catch (error) {
    console.error("Error in Falcon response generation:", error)
    return generateEnhancedFallbackResponse(text, analysis, context, learningMode)
  }
}

async function callHuggingFaceAPI(
  text: string,
  analysis: any,
  context: string[],
  learningMode: string,
): Promise<string | null> {
  const HF_API_TOKEN = process.env.HUGGINGFACE_API_TOKEN

  if (!HF_API_TOKEN) {
    console.log("Hugging Face API token not found, skipping API call")
    return null
  }

  console.log("[v0] Using Hugging Face API with token:", HF_API_TOKEN.substring(0, 10) + "...")

  try {
    const response = await fetch("https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_API_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: createEnhancedFalconPrompt(text, analysis, context, learningMode),
        parameters: {
          max_new_tokens: learningMode === "advanced" ? 300 : 200,
          temperature: learningMode === "casual" ? 0.8 : 0.7,
          do_sample: true,
          top_p: 0.9,
          repetition_penalty: 1.1,
          return_full_text: false, // Only return generated text, not the prompt
        },
        options: {
          wait_for_model: true, // Wait if model is loading
          use_cache: false, // Get fresh responses
        },
      }),
    })

    console.log("[v0] HF API Response status:", response.status)
    console.log("[v0] HF API Response headers:", Object.fromEntries(response.headers.entries()))

    if (!response.ok) {
      const errorText = await response.text()
      console.error("[v0] Hugging Face API error details:", {
        status: response.status,
        statusText: response.statusText,
        body: errorText,
        tokenFormat: HF_API_TOKEN.startsWith("hf_") ? "Valid format" : "Invalid format - should start with hf_",
      })

      if (response.status === 403) {
        console.error(
          "[v0] 403 Error - Possible causes: Invalid token, insufficient permissions, or model access denied",
        )
      } else if (response.status === 503) {
        console.error("[v0] 503 Error - Model is loading, will retry with wait_for_model")
      }

      return null
    }

    const data = await response.json()
    console.log("[v0] HF API Response data:", data)

    if (data.error) {
      console.error("[v0] Hugging Face API error in response:", data.error)
      return null
    }

    if (Array.isArray(data) && data[0]?.generated_text) {
      return data[0].generated_text.trim()
    } else if (data.generated_text) {
      return data.generated_text.trim()
    }

    return null
  } catch (error) {
    console.error("[v0] Error calling Hugging Face API:", error)
    return null
  }
}

function createEnhancedFalconPrompt(text: string, analysis: any, context: string[], learningMode: string): string {
  let prompt = "### Instruction:\nYou are an expert English language assistant. "

  // Adjust behavior based on learning mode
  switch (learningMode) {
    case "casual":
      prompt +=
        "Provide helpful, friendly responses about English language topics. Be conversational and encouraging.\n"
      break
    case "focused":
      prompt += "Focus on educational content with clear explanations and practical learning opportunities.\n"
      break
    case "advanced":
      prompt +=
        "Provide detailed linguistic analysis, advanced grammar concepts, and comprehensive explanations with examples.\n"
      break
  }

  // Add conversation context
  if (context.length > 0) {
    prompt += "\nPrevious conversation:\n"
    context.slice(-3).forEach((msg, idx) => {
      prompt += `${idx + 1}. ${msg}\n`
    })
  }

  // Add spaCy analysis context
  if (analysis) {
    prompt += "\nText Analysis Results:\n"
    if (analysis.entities && analysis.entities.length > 0) {
      prompt += `- Named entities: ${analysis.entities.map((e: any) => `${e.text} (${e.label})`).join(", ")}\n`
    }
    if (analysis.statistics) {
      prompt += `- Text stats: ${analysis.statistics.num_sentences} sentences, ${analysis.statistics.num_words} words\n`
    }
    if (analysis.grammar_issues && analysis.grammar_issues.length > 0) {
      prompt += `- Grammar issues found: ${analysis.grammar_issues.length}\n`
    }
  }

  prompt += `\n### Input:\n${text}\n\n### Response:\n`
  return prompt
}

function generateSuggestions(
  text: string,
  analysis: any,
  learningMode: string,
): Array<{ type: string; message: string; original?: string; suggested?: string }> {
  const suggestions = []

  if (!analysis) return suggestions

  // Grammar suggestions
  if (analysis.grammar_issues) {
    analysis.grammar_issues.forEach((issue: any) => {
      suggestions.push({
        type: "Grammar",
        message: issue.message,
        original: issue.original,
        suggested: issue.suggested,
      })
    })
  }

  // Style suggestions based on text analysis
  if (analysis.statistics) {
    const avgWordsPerSentence = analysis.statistics.num_words / analysis.statistics.num_sentences

    if (avgWordsPerSentence > 25) {
      suggestions.push({
        type: "Style",
        message: "Consider breaking long sentences into shorter ones for better readability.",
      })
    }

    if (analysis.statistics.num_stop_words / analysis.statistics.num_words > 0.5) {
      suggestions.push({
        type: "Vocabulary",
        message: "Try using more descriptive words to make your writing more engaging.",
      })
    }
  }

  // Learning mode specific suggestions
  if (learningMode === "focused" || learningMode === "advanced") {
    if (analysis.tokens) {
      const passiveVoice = analysis.tokens.filter(
        (token: any) => token.dep === "auxpass" || (token.pos === "VERB" && token.tag === "VBN"),
      )

      if (passiveVoice.length > 0) {
        suggestions.push({
          type: "Style",
          message: "Consider using active voice instead of passive voice for more direct communication.",
        })
      }
    }
  }

  return suggestions.slice(0, 3) // Limit to 3 suggestions
}

function generateLearningPoints(
  text: string,
  analysis: any,
  learningMode: string,
): Array<{ concept: string; explanation: string; examples: string[] }> {
  const learningPoints = []

  if (learningMode === "casual") return learningPoints

  if (!analysis) return learningPoints

  // Grammar concepts
  if (analysis.tokens) {
    const verbTenses = analysis.tokens.filter((token: any) => token.pos === "VERB")
    if (verbTenses.length > 0) {
      const tenseTypes = [...new Set(verbTenses.map((v: any) => v.tag))]
      if (tenseTypes.length > 1) {
        learningPoints.push({
          concept: "Verb Tenses",
          explanation: "Your text uses multiple verb tenses. Consistent tense usage helps maintain clarity.",
          examples: ["Past: walked", "Present: walks", "Future: will walk"],
        })
      }
    }
  }

  // Entity recognition learning
  if (analysis.entities && analysis.entities.length > 0) {
    const entityTypes = [...new Set(analysis.entities.map((e: any) => e.label))]
    learningPoints.push({
      concept: "Named Entity Recognition",
      explanation: "Named entities are specific names of people, places, organizations, etc.",
      examples: analysis.entities.slice(0, 3).map((e: any) => `${e.text} (${e.label})`),
    })
  }

  return learningPoints.slice(0, 2) // Limit to 2 learning points
}

function calculateLearningUpdate(text: string, analysis: any, learningMode: string): any {
  if (learningMode === "casual") return null

  const update: any = {}

  // Grammar progress
  if (analysis?.grammar_issues) {
    if (analysis.grammar_issues.length === 0) {
      update.grammar = Math.min(100, 75 + 2) // Increase by 2 points for good grammar
    }
  }

  // Vocabulary progress
  if (analysis?.statistics) {
    const uniqueWords = analysis.statistics.num_words - analysis.statistics.num_stop_words
    if (uniqueWords > 10) {
      update.vocabulary = Math.min(100, 60 + 1) // Increase by 1 point for rich vocabulary
    }
  }

  // Writing progress
  if (text.length > 100) {
    update.writing = Math.min(100, 80 + 1) // Increase by 1 point for substantial writing
  }

  return Object.keys(update).length > 0 ? update : null
}

async function callLocalFalcon(
  text: string,
  analysis: any,
  context: string[],
  learningMode: string,
): Promise<string | null> {
  try {
    // Call local Falcon server (if running)
    const response = await fetch("http://localhost:8001/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: createEnhancedFalconPrompt(text, analysis, context, learningMode),
        max_tokens: 200,
        temperature: 0.7,
      }),
    })

    if (!response.ok) {
      return null
    }

    const data = await response.json()
    return data.response || null
  } catch (error) {
    console.log("Local Falcon server not available, using fallback")
    return null
  }
}

function generateEnhancedFallbackResponse(
  text: string,
  analysis: any,
  context: string[],
  learningMode: string,
): string {
  const lowerText = text.toLowerCase()

  if (lowerText.includes("grammar") || lowerText.includes("correct") || hasObviousGrammarErrors(text)) {
    const correctedText = performBasicGrammarCorrection(text)

    if (correctedText !== text) {
      return `Here's the corrected version:\n\n**Original:** ${text}\n**Corrected:** ${correctedText}\n\nKey changes: ${getGrammarExplanation(text, correctedText)}`
    }

    if (analysis?.grammar_issues && analysis.grammar_issues.length > 0) {
      return `I found ${analysis.grammar_issues.length} potential grammar issue(s) in your text. ${analysis.grammar_issues[0].message} Would you like me to help you fix these issues?`
    }

    return "Your grammar looks good! If you have specific text you'd like me to check, please share it and I'll provide detailed corrections and explanations."
  }

  // Writing assistance
  if (lowerText.includes("write") || lowerText.includes("writing") || lowerText.includes("improve")) {
    const suggestions = []
    if (analysis?.statistics) {
      if (analysis.statistics.num_sentences === 1 && analysis.statistics.num_words > 20) {
        suggestions.push("Consider breaking this into shorter sentences for better readability.")
      }
      if (analysis.statistics.num_stop_words / analysis.statistics.num_words > 0.4) {
        suggestions.push("Try using more descriptive words to make your writing more engaging.")
      }
    }

    if (suggestions.length > 0) {
      return `Here are some writing suggestions for your text: ${suggestions.join(" ")} Would you like more specific feedback?`
    }
    return "I'd be happy to help improve your writing! I can assist with clarity, style, grammar, and structure. What specific aspect would you like to work on?"
  }

  // Analysis requests
  if (lowerText.includes("analyze") || lowerText.includes("analysis")) {
    if (analysis) {
      let response = "Here's my analysis of your text:\n\n"
      if (analysis.entities && analysis.entities.length > 0) {
        response += `• Found ${analysis.entities.length} named entities: ${analysis.entities
          .slice(0, 3)
          .map((e: any) => e.text)
          .join(", ")}\n`
      }
      if (analysis.statistics) {
        response += `• Text statistics: ${analysis.statistics.num_sentences} sentences, ${analysis.statistics.num_words} words\n`
      }
      if (analysis.grammar_issues && analysis.grammar_issues.length > 0) {
        response += `• Grammar: ${analysis.grammar_issues.length} potential issues detected\n`
      }
      response += "\nWould you like me to elaborate on any of these aspects?"
      return response
    }
    return "I can analyze text for various linguistic features including entities, grammar, style, and structure. Please provide the text you'd like me to analyze."
  }

  // Entity-related responses
  if (analysis?.entities && analysis.entities.length > 0) {
    const entityTypes = [...new Set(analysis.entities.map((e: any) => e.label))]
    return `I've identified several important elements in your text, including ${entityTypes.join(", ").toLowerCase()} entities. Your text appears to be about ${analysis.entities[0].text}. How can I help you work with this content?`
  }

  // Default helpful responses
  const responses = [
    "I'm here to help you with English language tasks! I can assist with grammar checking, writing improvement, text analysis, and language learning. What would you like to work on?",
    "I've processed your message and I'm ready to help! I can analyze text, check grammar, provide writing suggestions, or answer questions about English language usage.",
    "Thank you for your input! I specialize in helping with English language tasks including grammar, writing, analysis, and learning. How can I assist you today?",
    "I'm your English language assistant! I can help with grammar correction, writing enhancement, text analysis, and language questions. What specific help do you need?",
  ]

  return responses[Math.floor(Math.random() * responses.length)]
}

// Helper functions for grammar correction
function hasObviousGrammarErrors(text: string): boolean {
  const patterns = [
    /\bis\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i, // "is go", "is come", etc.
    /\bare\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i, // "are go", "are come", etc.
    /\bam\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i, // "am go", "am come", etc.
    /\b[a-z]+\s+is\s+go\b/i, // "someone is go"
    /\b[a-z]+\s+are\s+go\b/i, // "they are go"
    /\b(woman|man|boy|girl|person|student|teacher|doctor|child)\s+have\b/i, // singular subject + have
    /\bmany\s+[a-z]+(?!\s+(are|is|have|has|were|was))\b/i, // "many product" (missing plural)
    /\b[a-z]+\s+selling\s*$/i, // ending with "selling" (incomplete phrase)
    /\b(he|she|it)\s+have\b/i, // he/she/it have (should be has)
    /\b(I|you|we|they)\s+has\b/i, // I/you/we/they has (should be have)
    /\b[a-z]+\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\s+to\b/i, // "kofi go to school"
    /\b[a-z]+\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\s+(home|school|work|store|market)\b/i, // "kofi go school"
    /\b(kofi|ama|kwame|john|mary|peter|sarah|david|jane)\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i, // proper names + base verb
  ]

  return patterns.some((pattern) => pattern.test(text))
}

function performBasicGrammarCorrection(text: string): string {
  let corrected = text

  // Fix common verb form errors
  corrected = corrected.replace(/\bis\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/gi, (match, verb) => {
    return `is ${getCorrectVerbForm(verb, "present_continuous")}`
  })

  corrected = corrected.replace(/\bare\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/gi, (match, verb) => {
    return `are ${getCorrectVerbForm(verb, "present_continuous")}`
  })

  corrected = corrected.replace(/\bam\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/gi, (match, verb) => {
    return `am ${getCorrectVerbForm(verb, "present_continuous")}`
  })

  // Fix "name + base verb" -> "name + goes/is going"
  corrected = corrected.replace(
    /\b([a-z]+)\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)(\s+to\s+[a-z]+|\s+[a-z]+|$)/gi,
    (match, subject, verb, rest) => {
      // Use simple present for habitual actions, present continuous for ongoing
      const correctedVerb = getCorrectVerbForm(verb, "simple_present")
      return `${subject} ${correctedVerb}${rest}`
    },
  )

  // Fix singular subjects with "have" -> "has"
  corrected = corrected.replace(
    /\b(woman|man|boy|girl|person|student|teacher|doctor|child)\s+have\b/gi,
    (match, subject) => {
      return `${subject} has`
    },
  )

  // Capitalize proper nouns (common names)
  const properNouns = ["kofi", "ama", "kwame", "akosua", "john", "mary", "peter", "sarah", "david", "jane"]
  properNouns.forEach((name) => {
    const regex = new RegExp(`\\b${name}\\b`, "gi")
    corrected = corrected.replace(regex, name.charAt(0).toUpperCase() + name.slice(1).toLowerCase())
  })

  // Capitalize first letter of sentence
  corrected = corrected.charAt(0).toUpperCase() + corrected.slice(1)

  // Fix double spaces
  corrected = corrected.replace(/\s+/g, " ").trim()

  return corrected
}

function makePlural(noun: string): string {
  const lowerNoun = noun.toLowerCase()

  // Common irregular plurals
  const irregulars: { [key: string]: string } = {
    child: "children",
    person: "people",
    man: "men",
    woman: "women",
    foot: "feet",
    tooth: "teeth",
    mouse: "mice",
    goose: "geese",
  }

  if (irregulars[lowerNoun]) {
    return irregulars[lowerNoun]
  }

  // Regular plural rules
  if (
    lowerNoun.endsWith("s") ||
    lowerNoun.endsWith("sh") ||
    lowerNoun.endsWith("ch") ||
    lowerNoun.endsWith("x") ||
    lowerNoun.endsWith("z")
  ) {
    return noun + "es"
  }

  if (lowerNoun.endsWith("y") && !"aeiou".includes(lowerNoun[lowerNoun.length - 2])) {
    return noun.slice(0, -1) + "ies"
  }

  if (lowerNoun.endsWith("f")) {
    return noun.slice(0, -1) + "ves"
  }

  if (lowerNoun.endsWith("fe")) {
    return noun.slice(0, -2) + "ves"
  }

  // Default: add 's'
  return noun + "s"
}

function getCorrectVerbForm(verb: string, tense: string): string {
  const verbForms: { [key: string]: { [key: string]: string } } = {
    go: { present_continuous: "going", simple_present: "goes" },
    come: { present_continuous: "coming", simple_present: "comes" },
    run: { present_continuous: "running", simple_present: "runs" },
    jump: { present_continuous: "jumping", simple_present: "jumps" },
    walk: { present_continuous: "walking", simple_present: "walks" },
    play: { present_continuous: "playing", simple_present: "plays" },
    eat: { present_continuous: "eating", simple_present: "eats" },
    drink: { present_continuous: "drinking", simple_present: "drinks" },
    sleep: { present_continuous: "sleeping", simple_present: "sleeps" },
    work: { present_continuous: "working", simple_present: "works" },
  }

  return verbForms[verb.toLowerCase()]?.[tense] || verb
}

function getGrammarExplanation(original: string, corrected: string): string {
  const explanations = []

  if (original.toLowerCase() !== corrected.toLowerCase()) {
    explanations.push("capitalized proper nouns")
  }

  if (/\bis\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i.test(original)) {
    explanations.push("changed to present continuous tense (is + verb-ing)")
  }

  if (/\bare\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i.test(original)) {
    explanations.push("changed to present continuous tense (are + verb-ing)")
  }

  if (
    /\b[a-z]+\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i.test(original) &&
    !/\b(is|are|am)\s+(go|come|run|jump|walk|play|eat|drink|sleep|work)\b/i.test(original)
  ) {
    explanations.push("added proper verb form (third person singular)")
  }

  if (/\b(woman|man|boy|girl|person|student|teacher|doctor|child)\s+have\b/i.test(original)) {
    explanations.push("fixed subject-verb agreement (singular subject takes 'has')")
  }

  if (/\bmany\s+[a-z]+(?!\s+(are|is|have|has|were|was))\b/i.test(original)) {
    explanations.push("made noun plural after 'many'")
  }

  if (/\b[a-z]+\s+selling\s*$/i.test(original)) {
    explanations.push("improved phrase structure")
  }

  return explanations.length > 0 ? explanations.join(", ") : "improved sentence structure"
}

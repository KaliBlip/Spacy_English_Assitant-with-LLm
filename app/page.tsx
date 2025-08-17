"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Mic, MicOff, Send, Volume2, VolumeX, BookOpen, Target, Lightbulb, BarChart3 } from "lucide-react"
import type { SpeechRecognition } from "web-speech-api"

interface Message {
  id: string
  type: "user" | "assistant"
  content: string
  timestamp: Date
  entities?: Array<{ text: string; label: string }>
  suggestions?: Array<{ type: string; message: string; original?: string; suggested?: string }>
  learningPoints?: Array<{ concept: string; explanation: string; examples: string[] }>
}

interface LearningProgress {
  grammar: number
  vocabulary: number
  writing: number
  comprehension: number
}

export default function EnglishAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isListening, setIsListening] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const [activeTab, setActiveTab] = useState("chat")
  const [learningProgress, setLearningProgress] = useState<LearningProgress>({
    grammar: 75,
    vocabulary: 60,
    writing: 80,
    comprehension: 70,
  })

  const [conversationContext, setConversationContext] = useState<string[]>([])
  const [learningMode, setLearningMode] = useState<"casual" | "focused" | "advanced">("casual")

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)

  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== "undefined" && "webkitSpeechRecognition" in window) {
      const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = false
      recognitionRef.current.interimResults = false
      recognitionRef.current.lang = "en-US"

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setInput(transcript)
        setIsListening(false)
      }

      recognitionRef.current.onerror = () => {
        setIsListening(false)
      }

      recognitionRef.current.onend = () => {
        setIsListening(false)
      }
    }
  }, [])

  const startListening = () => {
    if (recognitionRef.current) {
      setIsListening(true)
      recognitionRef.current.start()
    }
  }

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      setIsListening(false)
    }
  }

  const speak = (text: string) => {
    if (!voiceEnabled || !("speechSynthesis" in window)) return

    setIsSpeaking(true)
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.onend = () => setIsSpeaking(false)
    speechSynthesis.speak(utterance)
  }

  const stopSpeaking = () => {
    if ("speechSynthesis" in window) {
      speechSynthesis.cancel()
      setIsSpeaking(false)
    }
  }

  const processMessage = async (text: string) => {
    if (!text.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: text,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsProcessing(true)

    // Update conversation context
    setConversationContext((prev) => [...prev.slice(-4), text]) // Keep last 5 messages for context

    try {
      // Enhanced processing with context and learning mode
      const response = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          context: conversationContext,
          learningMode,
          previousMessages: messages.slice(-3), // Send last 3 messages for context
        }),
      })

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: data.response,
        timestamp: new Date(),
        entities: data.entities,
        suggestions: data.suggestions,
        learningPoints: data.learningPoints,
      }

      setMessages((prev) => [...prev, assistantMessage])

      // Update learning progress based on interaction
      if (data.learningUpdate) {
        setLearningProgress((prev) => ({
          ...prev,
          ...data.learningUpdate,
        }))
      }

      // Speak the response if voice is enabled
      if (voiceEnabled) {
        speak(data.response)
      }
    } catch (error) {
      console.error("Error processing message:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: "Sorry, I encountered an error processing your request.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsProcessing(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    processMessage(input)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4">
      <div className="max-w-6xl mx-auto">
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-center flex items-center justify-center gap-2">
              🤖 English AI Assistant
              <div className="flex gap-2 ml-4">
                <Button variant="outline" size="sm" onClick={() => setVoiceEnabled(!voiceEnabled)}>
                  {voiceEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                </Button>
                {isSpeaking && (
                  <Button variant="outline" size="sm" onClick={stopSpeaking}>
                    Stop Speaking
                  </Button>
                )}
              </div>
            </CardTitle>
          </CardHeader>
        </Card>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="chat" className="flex items-center gap-2">
              <Send className="h-4 w-4" />
              Chat
            </TabsTrigger>
            <TabsTrigger value="learning" className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Learning
            </TabsTrigger>
            <TabsTrigger value="practice" className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              Practice
            </TabsTrigger>
            <TabsTrigger value="progress" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Progress
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chat">
            <Card className="mb-6 h-96 overflow-hidden">
              <CardContent className="p-4 h-full overflow-y-auto">
                {messages.length === 0 ? (
                  <div className="text-center text-muted-foreground mt-20">
                    <p className="text-lg mb-2">Welcome to your English AI Assistant!</p>
                    <p>I can help you with grammar, writing, language analysis, and more.</p>
                    <p className="text-sm mt-4">Try asking me to analyze text, check grammar, or help with writing.</p>

                    <div className="mt-6 flex justify-center gap-2">
                      <Button
                        variant={learningMode === "casual" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setLearningMode("casual")}
                      >
                        Casual Mode
                      </Button>
                      <Button
                        variant={learningMode === "focused" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setLearningMode("focused")}
                      >
                        Learning Mode
                      </Button>
                      <Button
                        variant={learningMode === "advanced" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setLearningMode("advanced")}
                      >
                        Advanced Mode
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[80%] p-3 rounded-lg ${
                            message.type === "user" ? "bg-blue-500 text-white" : "bg-gray-100 dark:bg-gray-700"
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{message.content}</p>

                          {message.entities && message.entities.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {message.entities.map((entity, idx) => (
                                <Badge key={idx} variant="secondary" className="text-xs">
                                  {entity.text} ({entity.label})
                                </Badge>
                              ))}
                            </div>
                          )}

                          {message.suggestions && message.suggestions.length > 0 && (
                            <div className="mt-3 space-y-2">
                              <p className="text-sm font-semibold flex items-center gap-1">
                                <Lightbulb className="h-3 w-3" />
                                Suggestions:
                              </p>
                              {message.suggestions.map((suggestion, idx) => (
                                <div key={idx} className="text-xs bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
                                  <p className="font-medium">{suggestion.type}:</p>
                                  <p>{suggestion.message}</p>
                                  {suggestion.original && suggestion.suggested && (
                                    <p className="mt-1">
                                      <span className="line-through text-red-600">{suggestion.original}</span>
                                      {" → "}
                                      <span className="text-green-600">{suggestion.suggested}</span>
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                          {message.learningPoints && message.learningPoints.length > 0 && (
                            <div className="mt-3 space-y-2">
                              <p className="text-sm font-semibold flex items-center gap-1">
                                <BookOpen className="h-3 w-3" />
                                Learning Points:
                              </p>
                              {message.learningPoints.map((point, idx) => (
                                <div key={idx} className="text-xs bg-green-50 dark:bg-green-900/20 p-2 rounded">
                                  <p className="font-medium">{point.concept}</p>
                                  <p>{point.explanation}</p>
                                  {point.examples.length > 0 && (
                                    <p className="mt-1 italic">Examples: {point.examples.join(", ")}</p>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                          <p className="text-xs opacity-70 mt-1">{message.timestamp.toLocaleTimeString()}</p>
                        </div>
                      </div>
                    ))}
                    {isProcessing && (
                      <div className="flex justify-start">
                        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                          <div className="flex items-center gap-2">
                            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                            <span>Processing...</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="learning">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BookOpen className="h-5 w-5" />
                    Grammar Lessons
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <h4 className="font-medium">Current Focus: Verb Tenses</h4>
                    <p className="text-sm text-muted-foreground">
                      Master the perfect tenses with interactive examples and practice exercises.
                    </p>
                    <Button size="sm" className="w-full">
                      Start Lesson
                    </Button>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-medium">Subject-Verb Agreement</h4>
                    <p className="text-sm text-muted-foreground">
                      Learn to match subjects with their corresponding verbs correctly.
                    </p>
                    <Button size="sm" variant="outline" className="w-full bg-transparent">
                      Review
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Writing Workshops
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <h4 className="font-medium">Essay Structure</h4>
                    <p className="text-sm text-muted-foreground">
                      Learn to organize your thoughts into compelling essays.
                    </p>
                    <Button size="sm" className="w-full">
                      Start Workshop
                    </Button>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-medium">Business Writing</h4>
                    <p className="text-sm text-muted-foreground">
                      Professional communication skills for the workplace.
                    </p>
                    <Button size="sm" variant="outline" className="w-full bg-transparent">
                      Explore
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="practice">
            <Card>
              <CardHeader>
                <CardTitle>Practice Exercises</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-3">
                  <Button className="h-20 flex flex-col items-center justify-center gap-2">
                    <Target className="h-6 w-6" />
                    Grammar Quiz
                  </Button>
                  <Button
                    variant="outline"
                    className="h-20 flex flex-col items-center justify-center gap-2 bg-transparent"
                  >
                    <BookOpen className="h-6 w-6" />
                    Vocabulary Builder
                  </Button>
                  <Button
                    variant="outline"
                    className="h-20 flex flex-col items-center justify-center gap-2 bg-transparent"
                  >
                    <Lightbulb className="h-6 w-6" />
                    Writing Prompts
                  </Button>
                </div>

                <div className="border rounded-lg p-4">
                  <h4 className="font-medium mb-2">Daily Challenge</h4>
                  <p className="text-sm text-muted-foreground mb-4">
                    Identify and correct the grammar error in this sentence:
                  </p>
                  <p className="italic bg-gray-50 dark:bg-gray-800 p-3 rounded">
                    "The team are working on their project and hopes to finish it by tomorrow."
                  </p>
                  <Button size="sm" className="mt-3">
                    Submit Answer
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="progress">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Learning Progress</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Grammar</span>
                      <span>{learningProgress.grammar}%</span>
                    </div>
                    <Progress value={learningProgress.grammar} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Vocabulary</span>
                      <span>{learningProgress.vocabulary}%</span>
                    </div>
                    <Progress value={learningProgress.vocabulary} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Writing</span>
                      <span>{learningProgress.writing}%</span>
                    </div>
                    <Progress value={learningProgress.writing} />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Comprehension</span>
                      <span>{learningProgress.comprehension}%</span>
                    </div>
                    <Progress value={learningProgress.comprehension} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Achievements</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-3 p-2 bg-green-50 dark:bg-green-900/20 rounded">
                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                      <Target className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">Grammar Master</p>
                      <p className="text-xs text-muted-foreground">Completed 10 grammar exercises</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                      <BookOpen className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">Word Wizard</p>
                      <p className="text-xs text-muted-foreground">Learned 25 new vocabulary words</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>

        <Card>
          <CardContent className="p-4">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <div className="flex-1 relative">
                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message or use voice input..."
                  className="min-h-[60px] pr-12"
                  disabled={isProcessing}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-2"
                  onClick={isListening ? stopListening : startListening}
                  disabled={isProcessing}
                >
                  {isListening ? <MicOff className="h-4 w-4 text-red-500" /> : <Mic className="h-4 w-4" />}
                </Button>
              </div>
              <Button type="submit" disabled={!input.trim() || isProcessing}>
                <Send className="h-4 w-4" />
              </Button>
            </form>
            {isListening && <p className="text-sm text-blue-500 mt-2 text-center">🎤 Listening... Speak now</p>}

            <div className="mt-2 text-center">
              <Badge variant="outline" className="text-xs">
                {learningMode === "casual" && "Casual Mode - Relaxed conversation"}
                {learningMode === "focused" && "Learning Mode - Educational focus"}
                {learningMode === "advanced" && "Advanced Mode - Detailed analysis"}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
